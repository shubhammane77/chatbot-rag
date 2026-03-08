import os
import uuid
from typing import Iterable, List, Optional

from langchain_core.documents import Document

try:
    from elasticsearch import Elasticsearch, helpers  # type: ignore[import]

    ELASTICSEARCH_IMPORT_ERROR = None
except ImportError as import_error:  # pragma: no cover - optional dependency
    Elasticsearch = None  # type: ignore[assignment]
    helpers = None  # type: ignore[assignment]
    ELASTICSEARCH_IMPORT_ERROR = import_error

DEFAULT_INDEX_NAME = "testing"


def _require_elasticsearch():
    if Elasticsearch is None:
        raise ImportError(
            "elasticsearch Python client is required but not installed."
        ) from ELASTICSEARCH_IMPORT_ERROR


def normalize_page_value(page_value: Optional[object]) -> Optional[int]:
    try:
        return int(page_value) if page_value is not None else None
    except (TypeError, ValueError):
        return None


def generate_document_id(document: Document) -> str:
    source = document.metadata.get("source", "unknown_source")
    page = normalize_page_value(document.metadata.get("page"))
    page_part = page if page is not None else -1
    base_identifier = f"{source}:{page_part}"
    stable_uuid = uuid.uuid5(uuid.NAMESPACE_URL, base_identifier)
    return str(stable_uuid)


def build_elasticsearch_client() -> "Elasticsearch":
    _require_elasticsearch()

    host = os.getenv("ELASTICSEARCH_HOST", "")
    api_key = os.getenv("ELASTICSEARCH_API_KEY")
    username = os.getenv("ELASTICSEARCH_USERNAME", "")
    password = os.getenv("ELASTICSEARCH_PASSWORD", "")
    verify_certs = os.getenv("ELASTICSEARCH_VERIFY_CERTS", "false").lower() == "true"

    client_kwargs = {"hosts": [host], "verify_certs": verify_certs}

    if api_key:
        client_kwargs["api_key"] = api_key
    elif username and password:
        client_kwargs["basic_auth"] = (username, password)

    return Elasticsearch(**client_kwargs)


def ensure_index(client: "Elasticsearch", index_name: str):
    if client.indices.exists(index=index_name):
        return

    mappings = {
        "mappings": {
            "properties": {
                "doc_id": {"type": "keyword"},
                "source": {"type": "keyword"},
                "page": {"type": "integer"},
                "content": {"type": "text"},
                "metadata": {"type": "object", "enabled": True},
            }
        }
    }
    client.indices.create(index=index_name, **mappings)


def persist_documents_to_elasticsearch(
    client: "Elasticsearch",
    documents: List[Document],
    index_name: Optional[str] = None,
):
    _require_elasticsearch()
    if helpers is None:
        raise ImportError(
            "elasticsearch.helpers is required but not installed."
        ) from ELASTICSEARCH_IMPORT_ERROR

    chosen_index = index_name or os.getenv("ELASTICSEARCH_INDEX", DEFAULT_INDEX_NAME)
    ensure_index(client, chosen_index)

    actions = []
    for document in documents:
        doc_id = generate_document_id(document)
        page = normalize_page_value(document.metadata.get("page"))
        document.metadata["page"] = page
        document.metadata["doc_id"] = doc_id
        actions.append(
            {
                "_index": chosen_index,
                "_id": doc_id,
                "_source": {
                    "doc_id": doc_id,
                    "source": document.metadata.get("source"),
                    "page": page,
                    "content": document.page_content,
                    "metadata": document.metadata,
                },
            }
        )

    if actions:
        helpers.bulk(client, actions)


def _resolve_index_name(index_name: Optional[str]) -> str:
    return index_name or os.getenv("ELASTICSEARCH_INDEX", DEFAULT_INDEX_NAME)


def search_elasticsearch_documents(
    client: "Elasticsearch",
    query: str,
    *,
    index_name: Optional[str] = None,
    size: int = 5,
    fields: Optional[Iterable[str]] = None,
):
    _require_elasticsearch()
    chosen_index = _resolve_index_name(index_name)

    if not client.indices.exists(index=chosen_index):
        raise ValueError(f"Elasticsearch index '{chosen_index}' does not exist.")

    searchable_fields = list(fields) if fields else ["content", "metadata.*", "source"]
    multi_match_query = {
        "query": query,
        "fields": searchable_fields,
        "lenient": True,
    }
    response = client.search(
        index=chosen_index,
        query={"multi_match": multi_match_query},
        size=size,
    )

    hits = response.get("hits", {}).get("hits", [])
    results = []
    for hit in hits:
        source_doc = hit.get("_source", {})
        results.append(
            {
                "doc_id": source_doc.get("doc_id"),
                "source": source_doc.get("source"),
                "page": source_doc.get("page"),
                "score": hit.get("_score"),
                "content": source_doc.get("content"),
                "metadata": source_doc.get("metadata"),
            }
        )
    return results

