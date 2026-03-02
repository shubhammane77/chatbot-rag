import os
from dotenv import load_dotenv

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

