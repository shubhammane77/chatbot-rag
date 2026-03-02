import os
from typing import Optional, Tuple

from langchain_core.language_models import BaseChatModel

from logs.logging_utils import log_debug
from utils.llm_factory import build_llm
from utils.rate_limiters import get_shared_llm_rate_limiter


class AnswerJudge:
    """
    LLM-based judge. Uses OpenRouter when OPENROUTER_API_KEY is set, otherwise Gemini.
    """

    def __init__(
        self,
        *,
        enabled: Optional[bool] = None,
        model_name: Optional[str] = None,
    ):
        if enabled is None:
            enabled = os.getenv("ENABLE_LLM_JUDGE", "true").lower() == "true"
        self.enabled = enabled
        self._llm: Optional[BaseChatModel] = None
        self._last_prompt: Optional[str] = None

        if not self.enabled:
            return

        try:
            rate_limiter = get_shared_llm_rate_limiter()
            self._llm = build_llm(
                model_name=model_name,
                temperature=0,
                rate_limiter=rate_limiter,
            )
        except Exception as error:  # pragma: no cover - network/config dependent
            log_debug(f"AnswerJudge initialization failed: {error}", force=True)
            self.enabled = False
            self._llm = None

    def evaluate(
        self, question: str, expected_answer: str, actual_answer: str
    ) -> Optional[Tuple[bool, str]]:
        if not self.enabled or self._llm is None:
            return None

        prompt = f"""You are a strict evaluator. Determine if the candidate answer is correct
given the question and reference answer.

Question: {question}
Reference answer: {expected_answer}
Candidate answer: {actual_answer}

Reply with "YES" if the candidate answer is correct and includes the reference answer, otherwise "NO". Optionally include a brief reason."""

        print("\n[AnswerJudge] Prompt being sent to LLM:\n")
        print(prompt)
        log_debug(f"[AnswerJudge Prompt]\n{prompt}", force=True)
        self._last_prompt = prompt

        try:
            response = self._llm.invoke(prompt)
        except Exception as error:  # pragma: no cover
            log_debug(f"AnswerJudge invocation failed: {error}", force=True)
            return None

        text = (getattr(response, "content", "") or "").strip().upper()
        if text.startswith("YES"):
            return True, "LLM judge: answer accepted"
        if text.startswith("NO"):
            return False, "LLM judge: answer rejected"
        return None

    @property
    def last_prompt(self) -> Optional[str]:
        return self._last_prompt

