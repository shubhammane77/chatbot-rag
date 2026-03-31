import os
from typing import List, Optional

from langchain_core.callbacks import (
    BaseCallbackHandler,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from pydantic import PrivateAttr
from utils.llm_factory import build_llm

from logs.logging_utils import (
    consume_last_retrieval_summary,
    is_debug_enabled,
    log_debug,
    set_last_retrieval_summary,
)
from logs.rag_logger import rag_log
from utils.rate_limiters import get_shared_llm_rate_limiter

from ingestion.elastic.elasticsearch_utils import (
    build_elasticsearch_client,
    search_elasticsearch_documents,
)

DEFAULT_SEMANTIC_K = 2
DEFAULT_LEXICAL_K = 3


class PromptLoggingHandler(BaseCallbackHandler):
    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def on_llm_start(self, serialized, prompts, **kwargs):
        if not self.enabled:
            return
        lines = ["\n=== LLM Input Prompt ==="]
        for idx, prompt in enumerate(prompts, start=1):
            lines.append(f"[Prompt {idx}]\n{prompt}\n")
        lines.append("=== End Prompt ===")

        retrieval_summary = consume_last_retrieval_summary()
        if retrieval_summary:
            lines.append("=== Retrieval Summary ===")
            lines.append(retrieval_summary)
            lines.append("=== End Retrieval Summary ===")

        message = "\n".join(lines) + "\n"
        print(message)
        log_debug(message, force=True)


RRF_K = 60  # standard constant for Reciprocal Rank Fusion


class HybridRetriever(BaseRetriever):
    semantic_retriever: BaseRetriever
    semantic_k: int = DEFAULT_SEMANTIC_K
    lexical_k: int = DEFAULT_LEXICAL_K
    es_index: Optional[str] = None
    debug: bool = False
    fusion_mode: str = "concat"  # "concat" or "rrf"
    _es_client: Optional[object] = PrivateAttr(default=None)

    def __init__(self, **data):
        es_client = data.pop("es_client", None)
        super().__init__(**data)
        self._es_client = es_client

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        lexical_docs = self._get_lexical_documents(query)
        semantic_docs = self._get_semantic_documents(query)
        if self.fusion_mode == "rrf":
            return self._rrf_merge(lexical_docs, semantic_docs)
        return self._deduplicate_documents(lexical_docs + semantic_docs)

    def _doc_key(self, doc: Document) -> str:
        metadata = doc.metadata or {}
        doc_id = metadata.get("doc_id")
        return doc_id or f"{metadata.get('source')}:{metadata.get('page')}:{hash(doc.page_content)}"

    def _rrf_merge(self, lexical_docs: List[Document], semantic_docs: List[Document]) -> List[Document]:
        """Merge two ranked lists using Reciprocal Rank Fusion.

        Docs appearing in both sources receive additive score boosts,
        naturally surfacing cross-source agreement to the top.
        """
        scores: dict = {}
        doc_map: dict = {}

        for rank, doc in enumerate(lexical_docs):
            key = self._doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            doc_map[key] = doc

        for rank, doc in enumerate(semantic_docs):
            key = self._doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            if key not in doc_map:
                doc_map[key] = doc

        return [doc_map[k] for k in sorted(scores, key=scores.__getitem__, reverse=True)]

    def _get_semantic_documents(self, query: str) -> List[Document]:
        if self.semantic_retriever is None or self.semantic_k <= 0:
            return []
        documents = self.semantic_retriever.invoke(query)
        for document in documents:
            if document.metadata is None:
                document.metadata = {}
            document.metadata["retrieval_mode"] = "semantic"
        docs = documents[: self.semantic_k]
        rag_log.retrieval(query=query, mode="semantic", documents=docs)
        return docs

    def _get_lexical_documents(self, query: str) -> List[Document]:
        if self._es_client is None or self.lexical_k <= 0:
            return []
        try:
            results = search_elasticsearch_documents(
                self._es_client,
                query,
                index_name=self.es_index,
                size=self.lexical_k,
            )
        except Exception as error:
            if self.debug:
                error_message = f"[HybridRetriever] Elasticsearch search failed: {error}"
                print(error_message)
                log_debug(error_message, force=True)
                set_last_retrieval_summary(error_message)
            return []

        summary_lines = [
            f"Elasticsearch results (requested {self.lexical_k}, using first {len(results)}):"
        ]
        lexical_docs: List[Document] = []
        for idx, result in enumerate(results, start=1):
            score = result.get("score")
            summary_lines.append(
                f"{idx}. score={score} source={result.get('source')} page={result.get('page')} doc_id={result.get('doc_id')}"
            )
            content = result.get("content", "")
            metadata = dict(result.get("metadata") or {})
            metadata.setdefault("doc_id", result.get("doc_id"))
            metadata.setdefault("source", result.get("source"))
            metadata.setdefault("page", result.get("page"))
            metadata["score"] = score
            metadata["retrieval_mode"] = "lexical"
            lexical_docs.append(Document(page_content=content, metadata=metadata))

        summary = "\n".join(summary_lines)
        log_debug(summary, force=True)
        set_last_retrieval_summary(summary)
        rag_log.retrieval(query=query, mode="lexical", documents=lexical_docs)

        return lexical_docs

    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        deduped: List[Document] = []
        seen_keys: set = set()
        for doc in documents:
            key = self._doc_key(doc)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(doc)
        return deduped

    def __del__(self):
        if self._es_client is not None:
            try:
                self._es_client.close()
            except Exception:
                pass


CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_TOP_K = 2


class CrossEncoderReranker(BaseRetriever):
    """Wraps any retriever and re-scores its results with a cross-encoder.

    The cross-encoder jointly encodes (query, document) pairs, producing
    a relevance score that is far more accurate than the bi-encoder cosine
    similarity used by FAISS.  Only the top ``top_k`` documents are returned,
    filtering out the lowest-confidence results and reducing pool pollution.
    """

    inner: BaseRetriever
    model_name: str = CROSS_ENCODER_MODEL
    top_k: int = CROSS_ENCODER_TOP_K
    _model: Optional[object] = PrivateAttr(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        except Exception as exc:
            log_debug(f"[CrossEncoderReranker] Failed to load model: {exc}", force=True)
            self._model = None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        docs = self.inner.invoke(query)
        if not docs or self._model is None:
            return docs

        pairs = [(query, doc.page_content) for doc in docs]
        scores = self._model.predict(pairs)

        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        for score, doc in ranked:
            if doc.metadata is None:
                doc.metadata = {}
            doc.metadata["cross_encoder_score"] = float(score)

        return [doc for _, doc in ranked[: self.top_k]]


_RAG_PROMPT = ChatPromptTemplate.from_template(
    "Use the following pieces of context to answer the question at the end. "
    "If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n"
    "{context}\n\n"
    "Question: {question}\n"
    "Helpful Answer:"
)


def build_rag_chain(retriever: BaseRetriever, *, mode: str = "hybrid"):
    rate_limiter = get_shared_llm_rate_limiter()
    debug_mode = is_debug_enabled()
    llm = build_llm(
        temperature=0,
        rate_limiter=rate_limiter,
        callbacks=[PromptLoggingHandler(enabled=debug_mode)],
        verbose=debug_mode,
    )
    hybrid_retriever = create_hybrid_retriever(retriever, mode=mode, debug_override=debug_mode)

    def retrieve(inputs):
        docs = hybrid_retriever.invoke(inputs["query"])
        return {"query": inputs["query"], "source_documents": docs}

    def format_for_prompt(inputs):
        context = "\n\n".join(d.page_content for d in inputs["source_documents"])
        return {
            "context": context,
            "question": inputs["query"],
            "source_documents": inputs["source_documents"],
        }

    answer_chain = (
        {"context": lambda x: x["context"], "question": lambda x: x["question"]}
        | _RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return (
        RunnableLambda(retrieve)
        | RunnableLambda(format_for_prompt)
        | RunnablePassthrough.assign(result=answer_chain)
        | RunnableLambda(lambda x: {"result": x["result"], "source_documents": x["source_documents"]})
    )


def create_hybrid_retriever(
    semantic_retriever: BaseRetriever,
    *,
    mode: str = "hybrid",
    semantic_k: int = DEFAULT_SEMANTIC_K,
    lexical_k: int = DEFAULT_LEXICAL_K,
    debug_override: Optional[bool] = None,
) -> BaseRetriever:
    """Build a retriever for the given mode.

    Supported modes
    ---------------
    ``faiss_only``    — FAISS semantic search only
    ``es_only``       — Elasticsearch lexical search only
    ``hybrid``        — simple union of both sources (deduped)
    ``rrf``           — Reciprocal Rank Fusion of both sources
    ``cross_encoder`` — hybrid union re-ranked by a cross-encoder (top-k filtered)
    """
    if mode == "faiss_only":
        return _create_hybrid_retriever_internal(
            semantic_retriever=semantic_retriever,
            semantic_k=DEFAULT_SEMANTIC_K,
            lexical_k=0,
            fusion_mode="concat",
            debug_override=debug_override,
        )

    if mode == "es_only":
        return _create_hybrid_retriever_internal(
            semantic_retriever=semantic_retriever,
            semantic_k=0,
            lexical_k=DEFAULT_LEXICAL_K,
            fusion_mode="concat",
            debug_override=debug_override,
        )

    if mode == "rrf":
        return _create_hybrid_retriever_internal(
            semantic_retriever=semantic_retriever,
            semantic_k=semantic_k,
            lexical_k=lexical_k,
            fusion_mode="rrf",
            debug_override=debug_override,
        )

    if mode == "cross_encoder":
        base = _create_hybrid_retriever_internal(
            semantic_retriever=semantic_retriever,
            semantic_k=semantic_k,
            lexical_k=lexical_k,
            fusion_mode="concat",
            debug_override=debug_override,
        )
        return CrossEncoderReranker(inner=base)

    # default: "hybrid" — simple concat union
    return _create_hybrid_retriever_internal(
        semantic_retriever=semantic_retriever,
        semantic_k=semantic_k,
        lexical_k=lexical_k,
        fusion_mode="concat",
        debug_override=debug_override,
    )


def _create_hybrid_retriever_internal(
    *,
    semantic_retriever: BaseRetriever,
    semantic_k: int,
    lexical_k: int,
    fusion_mode: str = "concat",
    debug_override: Optional[bool],
) -> HybridRetriever:
    debug_mode = (
        debug_override
        if debug_override is not None
        else is_debug_enabled()
    )
    es_client = None
    try:
        es_client = build_elasticsearch_client()
    except Exception as error:
        if debug_mode:
            print(f"[HybridRetriever] Elasticsearch client unavailable: {error}")

    return HybridRetriever(
        semantic_retriever=semantic_retriever,
        es_client=es_client,
        semantic_k=semantic_k,
        lexical_k=lexical_k,
        es_index=os.getenv("ELASTICSEARCH_INDEX"),
        debug=debug_mode,
        fusion_mode=fusion_mode,
    )
