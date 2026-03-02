import os
from typing import Optional

from langchain_core.language_models import BaseChatModel


_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_HF_MODEL = "Qwen/Qwen2.5-72B-Instruct"
_DEFAULT_OPENROUTER_MODEL = "google/gemma-3-27b-it:free"
_DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"


def build_llm(
    *,
    model_name: Optional[str] = None,
    temperature: float = 0,
    rate_limiter=None,
    **kwargs,
) -> BaseChatModel:
    """Return a chat LLM.

    Provider priority (first key found wins):
      1. HuggingFace  — HF_API_KEY
      2. OpenRouter   — OPENROUTER_API_KEY
      3. Vertex AI    — GEMINI_API_KEY (AQ. OAuth token) + GENAI_PROJECT

    Override the model via HF_MODEL / OPENROUTER_MODEL / GEMINI_MODEL env vars,
    or pass *model_name* directly.
    """
    hf_key = os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if hf_key:
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        resolved_model = model_name or os.getenv("HF_MODEL", _DEFAULT_HF_MODEL)
        # HF Inference API requires temperature > 0
        hf_temperature = max(float(temperature), 0.01)
        endpoint = HuggingFaceEndpoint(
            repo_id=resolved_model,
            huggingfacehub_api_token=hf_key,
            task="text-generation",
            temperature=hf_temperature,
            max_new_tokens=512,
        )
        chat_kwargs = {}
        if rate_limiter is not None:
            chat_kwargs["rate_limiter"] = rate_limiter
        return ChatHuggingFace(llm=endpoint, **chat_kwargs)

    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    if openrouter_key:
        from langchain_openai import ChatOpenAI

        resolved_model = model_name or os.getenv(
            "OPENROUTER_MODEL", _DEFAULT_OPENROUTER_MODEL
        )
        init_kwargs = dict(
            model=resolved_model,
            openai_api_key=openrouter_key,
            openai_api_base=_OPENROUTER_BASE_URL,
            temperature=temperature,
            **kwargs,
        )
        if rate_limiter is not None:
            init_kwargs["rate_limiter"] = rate_limiter
        return ChatOpenAI(**init_kwargs)

    from langchain_google_vertexai import ChatVertexAI

    resolved_model = model_name or os.getenv("GEMINI_MODEL", _DEFAULT_GEMINI_MODEL)
    project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GENAI_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    init_kwargs = dict(
        model=resolved_model,
        temperature=temperature,
        project=project,
        location=location,
        **kwargs,
    )
    if rate_limiter is not None:
        init_kwargs["rate_limiter"] = rate_limiter
    return ChatVertexAI(**init_kwargs)
