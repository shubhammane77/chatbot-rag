import os
from typing import List
from dotenv import load_dotenv

# _ALLOWED_CHUNKING = {"recursive", "fixed_size", "sliding_window"}
# _ALLOWED_RETRIEVAL = {"faiss_only", "es_only", "hybrid"}
# _ALLOWED_QUERY = {"none", "multi_query", "chain_of_thought", "step_back"}
# _ALLOWED_TABLES = {"with_tables", "without_tables"}

_ALLOWED_CHUNKING = {"recursive"}
_ALLOWED_RETRIEVAL = {"hybrid"}
_ALLOWED_QUERY = {"none"}
_ALLOWED_TABLES = {"with_tables"}

def load_eval_config() -> dict:
    """Return active evaluation dimensions parsed from env vars."""
    def _parse(env_var: str, allowed: set, default: List[str]) -> List[str]:
        raw = os.getenv(env_var, "").strip()
        if not raw:
            return list(default)
        values = [v.strip() for v in raw.split(",") if v.strip()]
        invalid = set(values) - allowed
        if invalid:
            raise ValueError(
                f"Invalid values for {env_var}: {invalid}. Allowed: {allowed}"
            )
        return values

    return {
        "chunking_strategies": _parse(
            "EVAL_CHUNKING_STRATEGIES",
            _ALLOWED_CHUNKING,
            sorted(_ALLOWED_CHUNKING),
        ),
        "retrieval_modes": _parse(
            "EVAL_RETRIEVAL_MODES",
            _ALLOWED_RETRIEVAL,
            sorted(_ALLOWED_RETRIEVAL),
        ),
        "query_strategies": _parse(
            "EVAL_QUERY_STRATEGIES",
            _ALLOWED_QUERY,
            sorted(_ALLOWED_QUERY),
        ),
        "table_extraction": _parse(
            "EVAL_TABLE_EXTRACTION",
            _ALLOWED_TABLES,
            ["with_tables"],
        ),
        "sample_count": int(os.getenv("EVAL_SAMPLE_COUNT", "20")),
    }


def load_environment_variables():
    load_dotenv()
    # Set LangSmith environment variables
    os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "true")
    os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
    os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "49036947708")
    os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    # LLM provider priority: HF → OpenRouter → Vertex AI
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
    os.environ["HF_MODEL"] = os.getenv("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct")
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
    os.environ["GROQ_MODEL"] = os.getenv("GROQ_MODEL", "llama3-8b-8192")
    os.environ["HF_REQUESTS_PER_MINUTE"] = os.getenv("HF_REQUESTS_PER_MINUTE", "10")
    os.environ["GROQ_REQUESTS_PER_MINUTE"] = os.getenv("GROQ_REQUESTS_PER_MINUTE", "30")
    os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY", "")
    os.environ["OPENROUTER_MODEL"] = os.getenv("OPENROUTER_MODEL", "google/gemma-3-27b-it:free")
    # Gemini / Vertex AI fallback
    os.environ["GEMINI_MODEL"] = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    os.environ["GOOGLE_CLOUD_PROJECT"] = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GENAI_PROJECT", "")
    os.environ["GOOGLE_CLOUD_LOCATION"] = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    # AQ. OAuth token: expose as GOOGLE_OAUTH_ACCESS_TOKEN so google-auth picks it up
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key.startswith("AQ."):
        os.environ["GOOGLE_OAUTH_ACCESS_TOKEN"] = gemini_key
    os.environ["DOCUMENT_PATH"] = os.getenv("DOCUMENT_PATH", "")
    os.environ["HF_EMBEDDING_MODEL"] = os.getenv("HF_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    os.environ["DEBUG_MODE"] = os.getenv("DEBUG_MODE", "false")

