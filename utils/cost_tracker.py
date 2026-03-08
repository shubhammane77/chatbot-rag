"""Global LLM cost/token tracker.

Usage
-----
The tracker is wired automatically via ``llm_factory.build_llm()``.
Call ``cost_tracker.print_summary()`` whenever you want a report, and
``cost_tracker.reset()`` to start a fresh accounting period.
"""
from __future__ import annotations

from threading import Lock
from typing import Any, Dict, List, Tuple
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult


# ---------------------------------------------------------------------------
# Pricing table  (USD per 1 000 000 tokens)
# ---------------------------------------------------------------------------
# fmt: off
_PRICING: Dict[str, Tuple[float, float]] = {
    # model-name-fragment            input $/M   output $/M
    # Groq-hosted Llama models
    "llama3-8b-8192":               (0.05,   0.08),
    "llama3-70b-8192":              (0.59,   0.79),
    "llama-3.1-8b-instant":         (0.05,   0.08),
    "llama-3.1-70b-versatile":      (0.59,   0.79),
    "llama-3.3-70b-versatile":      (0.59,   0.79),
    "llama-3.3-70b-specdec":        (0.59,   0.99),
    "llama3-groq-8b-8192-tool-use-preview":  (0.19, 0.19),
    "llama3-groq-70b-8192-tool-use-preview": (0.89, 0.89),
    # Groq other
    "mixtral-8x7b-32768":           (0.24,   0.24),
    "gemma2-9b-it":                 (0.20,   0.20),
    "gemma-7b-it":                  (0.07,   0.07),
    # OpenRouter free models (zero cost)
    "google/gemma-3-27b-it:free":   (0.00,   0.00),
    "meta-llama/llama-3.3-70b-instruct:free": (0.00, 0.00),
    # Gemini (Vertex AI)
    "gemini-2.0-flash":             (0.075,  0.30),
    "gemini-1.5-flash":             (0.075,  0.30),
    "gemini-1.5-pro":               (1.25,   5.00),
    # HuggingFace Inference API — typically free / pay-per-use; treat as zero
    "Qwen/Qwen2.5-72B-Instruct":    (0.00,   0.00),
}
# fmt: on


def _lookup_price(model: str) -> Tuple[float, float]:
    """Return (input_price_per_M, output_price_per_M) for *model*.

    Tries exact match first, then substring match so partial model names work.
    Returns (0, 0) if unknown.
    """
    if not model:
        return (0.0, 0.0)
    if model in _PRICING:
        return _PRICING[model]
    lower = model.lower()
    for key, price in _PRICING.items():
        if key.lower() in lower or lower in key.lower():
            return price
    return (0.0, 0.0)


def _cost_usd(input_tokens: int, output_tokens: int, model: str) -> float:
    inp_price, out_price = _lookup_price(model)
    return (input_tokens * inp_price + output_tokens * out_price) / 1_000_000


class _ModelStats:
    __slots__ = ("input_tokens", "output_tokens", "calls")

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0


class _TokenUsageCallback(BaseCallbackHandler):
    """LangChain callback that accumulates token counts and logs every LLM call."""

    def __init__(self, tracker: "CostTracker") -> None:
        super().__init__()
        self._tracker = tracker
        self._pending_prompts: Dict[UUID, str] = {}
        self._pending_models: Dict[UUID, str] = {}

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        model = (serialized.get("kwargs") or {}).get("model", "") or (
            serialized.get("name", "")
        )
        prompt_lines = []
        for msg_list in messages:
            for m in msg_list:
                role = getattr(m, "type", "unknown")
                prompt_lines.append(f"[{role}] {m.content}")
        self._pending_prompts[run_id] = "\n".join(prompt_lines)
        self._pending_models[run_id] = model

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        model = (serialized.get("kwargs") or {}).get("model", "") or (
            serialized.get("name", "")
        )
        self._pending_prompts[run_id] = "\n".join(prompts)
        self._pending_models[run_id] = model

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        input_tokens = 0
        output_tokens = 0
        response_text = ""

        # --- 1. Try usage_metadata on AIMessage (LangChain >= 0.2) ----------
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg is not None:
                    meta = getattr(msg, "usage_metadata", None)
                    if meta:
                        input_tokens += meta.get("input_tokens", 0)
                        output_tokens += meta.get("output_tokens", 0)
                    response_text += getattr(msg, "content", "") or ""
                else:
                    response_text += getattr(gen, "text", "") or ""

        # --- 2. Fallback: llm_output["token_usage"] (OpenAI-style) ----------
        if input_tokens == 0 and output_tokens == 0:
            llm_output: Dict = response.llm_output or {}
            token_usage: Dict = llm_output.get("token_usage", {})
            input_tokens = token_usage.get("prompt_tokens", 0)
            output_tokens = token_usage.get("completion_tokens", 0)

        prompt = self._pending_prompts.pop(run_id, "")
        model = self._pending_models.pop(run_id, "")

        # --- Log to JSONL (always-on) ----------------------------------------
        from logs.rag_logger import rag_log  # local import to avoid circular dep
        rag_log.llm_call(
            prompt=prompt,
            response=response_text,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        self._tracker._add(input_tokens, output_tokens, model)


class CostTracker:
    """Thread-safe accumulator for LLM token usage across all calls."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._input_tokens = 0
        self._output_tokens = 0
        self._call_count = 0
        self._per_model: Dict[str, _ModelStats] = {}
        self._callback = _TokenUsageCallback(self)

    def _add(self, input_tokens: int, output_tokens: int, model: str = "") -> None:
        with self._lock:
            self._input_tokens += input_tokens
            self._output_tokens += output_tokens
            self._call_count += 1
            key = model or "unknown"
            if key not in self._per_model:
                self._per_model[key] = _ModelStats()
            s = self._per_model[key]
            s.input_tokens += input_tokens
            s.output_tokens += output_tokens
            s.calls += 1

    @property
    def callback(self) -> _TokenUsageCallback:
        return self._callback

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            per_model = {}
            total_cost = 0.0
            for model, s in self._per_model.items():
                cost = _cost_usd(s.input_tokens, s.output_tokens, model)
                total_cost += cost
                per_model[model] = {
                    "calls": s.calls,
                    "input_tokens": s.input_tokens,
                    "output_tokens": s.output_tokens,
                    "total_tokens": s.input_tokens + s.output_tokens,
                    "cost_usd": cost,
                }
            return {
                "calls": self._call_count,
                "input_tokens": self._input_tokens,
                "output_tokens": self._output_tokens,
                "total_tokens": self._input_tokens + self._output_tokens,
                "cost_usd": total_cost,
                "per_model": per_model,
            }

    def reset(self) -> None:
        with self._lock:
            self._input_tokens = 0
            self._output_tokens = 0
            self._call_count = 0
            self._per_model.clear()

    def print_summary(self, label: str = "LLM Cost Summary") -> None:
        s = self.snapshot()
        sep = "-" * 50
        print(f"\n{sep}")
        print(f"  {label}")
        print(sep)
        print(f"  LLM calls     : {s['calls']}")
        print(f"  Input tokens  : {s['input_tokens']:,}")
        print(f"  Output tokens : {s['output_tokens']:,}")
        print(f"  Total tokens  : {s['total_tokens']:,}")
        print(f"  Est. cost     : ${s['cost_usd']:.6f}")

        if s["per_model"]:
            print(f"\n  {'Model':<40} {'Calls':>6} {'In tok':>9} {'Out tok':>9} {'Cost USD':>12}")
            print(f"  {'-'*40} {'------':>6} {'-'*9} {'-'*9} {'-'*12}")
            for model, m in sorted(s["per_model"].items()):
                known = _lookup_price(model) != (0.0, 0.0) or any(
                    k.lower() in model.lower() or model.lower() in k.lower()
                    for k in _PRICING
                )
                cost_str = f"${m['cost_usd']:.6f}" if known else "n/a"
                print(
                    f"  {model:<40} {m['calls']:>6,} {m['input_tokens']:>9,}"
                    f" {m['output_tokens']:>9,} {cost_str:>12}"
                )
        print(sep)


# Singleton — import and use this everywhere
cost_tracker = CostTracker()
