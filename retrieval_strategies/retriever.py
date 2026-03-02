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
from utils.rate_limiters import get_shared_llm_rate_limiter

from elastic.elasticsearch_utils import (
    build_elasticsearch_client,
    search_elasticsearch_documents,
)

DEFAULT_SEMANTIC_K = 2
DEFAULT_LEXICAL_K = 2


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


class HybridRetriever(BaseRetriever):
    semantic_retriever: BaseRetriever
    semantic_k: int = DEFAULT_SEMANTIC_K
    lexical_k: int = DEFAULT_LEXICAL_K
    es_index: Optional[str] = None
    debug: bool = False
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
        combined = lexical_docs  + semantic_docs
        return self._deduplicate_documents(combined)

    def _get_semantic_documents(self, query: str) -> List[Document]:
        if self.semantic_retriever is None:
            return []
        documents = self.semantic_retriever.invoke(query)
        for document in documents:
            if document.metadata is None:
                document.metadata = {}
            document.metadata["retrieval_mode"] = "semantic"
        if self.semantic_k > 0:
            return documents[: self.semantic_k]
        return documents

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

        return lexical_docs

    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        deduped: List[Document] = []
        seen_keys = set()
        for doc in documents:
            metadata = doc.metadata or {}
            doc_id = metadata.get("doc_id")
            key = doc_id or f"{metadata.get('source')}:{metadata.get('page')}:{hash(doc.page_content)}"
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


_RAG_PROMPT = ChatPromptTemplate.from_template(
    "Use the following pieces of context to answer the question at the end. "
    "If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n"
    "{context}\n\n"
    "Question: {question}\n"
    "Helpful Answer:"
)


def build_rag_chain(retriever: BaseRetriever):
    rate_limiter = get_shared_llm_rate_limiter()
    debug_mode = is_debug_enabled()
    llm = build_llm(
        temperature=0,
        rate_limiter=rate_limiter,
        callbacks=[PromptLoggingHandler(enabled=debug_mode)],
        verbose=debug_mode,
    )
    hybrid_retriever = create_hybrid_retriever(retriever, debug_override=debug_mode)

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
    semantic_k: int = DEFAULT_SEMANTIC_K,
    lexical_k: int = DEFAULT_LEXICAL_K,
    debug_override: Optional[bool] = None,
) -> BaseRetriever:
    return _create_hybrid_retriever_internal(
        semantic_retriever=semantic_retriever,
        semantic_k=semantic_k,
        lexical_k=lexical_k,
        debug_override=debug_override,
    )


def _create_hybrid_retriever_internal(
    *,
    semantic_retriever: BaseRetriever,
    semantic_k: int,
    lexical_k: int,
    debug_override: Optional[bool],
) -> BaseRetriever:
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
    )
