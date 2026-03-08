import os
from typing import Optional

from langchain_core.language_models import BaseChatModel
from utils.rate_limiters import get_provider_rate_limiter
from utils.cost_tracker import cost_tracker


_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_HF_MODEL = "Qwen/Qwen2.5-72B-Instruct"
_DEFAULT_GROQ_MODEL = "llama3-8b-8192"
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
      2. Groq         — GROQ_API_KEY
      3. OpenRouter   — OPENROUTER_API_KEY
      4. Vertex AI    — GEMINI_API_KEY (AQ. OAuth token) + GENAI_PROJECT

    Override the model via HF_MODEL / GROQ_MODEL / OPENROUTER_MODEL / GEMINI_MODEL
    env vars, or pass *model_name* directly.
    """
    extra_callbacks = kwargs.pop("callbacks", [])
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
        rl = rate_limiter if rate_limiter is not None else get_provider_rate_limiter("hf")
        return ChatHuggingFace(llm=endpoint, rate_limiter=rl, callbacks=[cost_tracker.callback] + extra_callbacks)

    groq_key = os.getenv("GROQ_API_KEY")

    if groq_key:
        from langchain_groq import ChatGroq

        resolved_model = model_name or os.getenv("GROQ_MODEL", _DEFAULT_GROQ_MODEL)
        rl = rate_limiter if rate_limiter is not None else get_provider_rate_limiter("groq")
        return ChatGroq(
            model=resolved_model,
            groq_api_key=groq_key,
            temperature=temperature,
            rate_limiter=rl,
            callbacks=[cost_tracker.callback] + extra_callbacks,
            **kwargs,
        )

    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    if openrouter_key:
        from langchain_openai import ChatOpenAI

        resolved_model = model_name or os.getenv(
            "OPENROUTER_MODEL", _DEFAULT_OPENROUTER_MODEL
        )
        rl = rate_limiter if rate_limiter is not None else get_provider_rate_limiter("openrouter")
        return ChatOpenAI(
            model=resolved_model,
            openai_api_key=openrouter_key,
            openai_api_base=_OPENROUTER_BASE_URL,
            temperature=temperature,
            rate_limiter=rl,
            callbacks=[cost_tracker.callback] + extra_callbacks,
            **kwargs,
        )

    from langchain_google_vertexai import ChatVertexAI

    resolved_model = model_name or os.getenv("GEMINI_MODEL", _DEFAULT_GEMINI_MODEL)
    project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GENAI_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    rl = rate_limiter if rate_limiter is not None else get_provider_rate_limiter("gemini")
    return ChatVertexAI(
        model=resolved_model,
        temperature=temperature,
        project=project,
        location=location,
        rate_limiter=rl,
        callbacks=[cost_tracker.callback] + extra_callbacks,
        **kwargs,
    )
