import csv
import itertools
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from langchain_core.documents import Document

from evaluator.judges import AnswerJudge
from evaluator.wrong_answer_recorder import WrongAnswerRecorder
from logs.logging_utils import is_debug_enabled, log_debug
from retrieval_strategies.retriever import build_rag_chain
from utils.cost_tracker import cost_tracker
from retrieval_strategies.chain_of_thought import decompose_query_with_cot
from retrieval_strategies.multi_query import generate_multi_queries
from retrieval_strategies.step_back import query_with_step_back


def _maybe_print_sources(response_dict):
    if not is_debug_enabled():
        return
    source_documents = response_dict.get("source_documents") or []
    if not source_documents:
        return

    lines = ["Retrieved Documents:"]
    for i, doc in enumerate(source_documents):
        metadata = doc.metadata or {}
        retrieval_mode = metadata.get("retrieval_mode", "semantic")
        source = metadata.get("source", "N/A")
        page = metadata.get("page", "N/A")
        snippet = (doc.page_content or "").strip().replace("\n", " ")[:300]
        lines.append(
            f"  Document {i+1} [{retrieval_mode}] (Source: {source}, Page: {page}):\n"
            f"    {snippet}"
        )

    message = "\n".join(lines)
    print(message)
    log_debug(message, force=True)

def chat_interface(vector_stores):
    selected_strategy_name = input("\nSelect a chunking strategy for chat (recursive, fixed_size, sliding_window): ").lower()
    while selected_strategy_name not in vector_stores:
        selected_strategy_name = input("Invalid strategy. Please choose from (recursive, fixed_size, sliding_window): ").lower()
            
    selected_vector_store = vector_stores[selected_strategy_name]
    retriever = selected_vector_store.as_retriever(search_kwargs={"k": 4})
    rag_chain = build_rag_chain(retriever)

    cost_tracker.reset()
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            cost_tracker.print_summary("Chat Session — LLM Cost Summary")
            break
        query_rewriting = input("\n Type m for multi-query generation, c for chain of thought prompting, s for step-by-step decomposition, or press Enter to run RAG directly: ")
        if query_rewriting.lower() == "m":
            multi_queries = generate_multi_queries(query)
            for mq in multi_queries:
                response_dict = rag_chain.invoke({"query": mq})
                print(f"\nAnswer for '{mq}':\n", response_dict["result"])
                _maybe_print_sources(response_dict)
        if query_rewriting.lower() == "c":
            steps = decompose_query_with_cot(query)
            print("\nDecomposed Steps:")
            for i, step in enumerate(steps, 1):
                print(f"{i}. {step}")
            print("\nNow running RAG for each step...")
            for step in steps:
                response_dict = rag_chain.invoke({"query": step})
                print(f"\nAnswer for '{step}':\n", response_dict["result"])
                _maybe_print_sources(response_dict)
        if query_rewriting.lower() == "s":
            steps = query_with_step_back(query)
            print("\nDecomposed Steps:")
            for i, step in enumerate(steps, 1):
                print(f"{i}. {step}")
            print("\nNow running RAG for each step...")
            for step in steps:
                response_dict = rag_chain.invoke({"query": step})
                print(f"\nAnswer for '{step}':\n", response_dict["result"])
                _maybe_print_sources(response_dict)
        else:
            response_dict = rag_chain.invoke({"query": query})
            print("\nAnswer:\n", response_dict["result"])
            _maybe_print_sources(response_dict)

def _run_query_with_strategy(rag_chain, query: str, strategy: str) -> str:
    """Run a query through the rag_chain using the given query strategy.

    Returns the concatenated answer string(s).
    """
    if strategy == "none":
        response_dict = rag_chain.invoke({"query": query})
        return response_dict["result"]

    if strategy == "multi_query":
        sub_queries = generate_multi_queries(query)
        if not sub_queries:
            response_dict = rag_chain.invoke({"query": query})
            return response_dict["result"]
        answers = []
        for sq in sub_queries:
            r = rag_chain.invoke({"query": sq})
            answers.append(r["result"])
        return "\n\n".join(answers)

    if strategy == "chain_of_thought":
        steps = decompose_query_with_cot(query)
        if not steps:
            response_dict = rag_chain.invoke({"query": query})
            return response_dict["result"]
        answers = []
        for step in steps:
            r = rag_chain.invoke({"query": step})
            answers.append(r["result"])
        return "\n\n".join(answers)

    if strategy == "step_back":
        steps = query_with_step_back(query)
        if not steps:
            response_dict = rag_chain.invoke({"query": query})
            return response_dict["result"]
        answers = []
        for step in steps:
            r = rag_chain.invoke({"query": step})
            answers.append(r["result"])
        return "\n\n".join(answers)

    # Fallback: treat as "none"
    response_dict = rag_chain.invoke({"query": query})
    return response_dict["result"]


def evaluate_strategies(
    all_vector_stores: Dict,
    eval_config: Optional[Dict] = None,
):
    """Run evaluation across all active dimension combinations.

    Parameters
    ----------
    all_vector_stores:
        Dict keyed by ``"{chunking_strategy}_{table_suffix}"`` where
        table_suffix is ``"with_tables"`` or ``"without_tables"``.
    eval_config:
        Output of ``config.load_eval_config()``.  When *None* the legacy
        single-key format ``{name: vector_store}`` is accepted for
        backwards-compatibility.
    """
    # --- backwards-compat: old callers pass plain {name: vs} dict ---
    if eval_config is None:
        _legacy_evaluate(all_vector_stores)
        return

    evaluation_data: List[dict] = []
    with open("FinQA/test/metadata.jsonl", "r") as f:
        for line in f:
            evaluation_data.append(json.loads(line))

    sample_count = eval_config.get("sample_count", 100)
    evaluation_data = evaluation_data[:sample_count]
    print(f"Loaded {len(evaluation_data)} evaluation samples (limit={sample_count}).")

    numeric_abs_tol = float(os.getenv("NUMERIC_ABSOLUTE_TOLERANCE", "1e-6"))
    numeric_rel_tol = float(os.getenv("NUMERIC_RELATIVE_TOLERANCE", "0.01"))
    judge = AnswerJudge()
    recorder = WrongAnswerRecorder()

    chunking_list = eval_config.get("chunking_strategies", [])
    retrieval_list = eval_config.get("retrieval_modes", [])
    query_list = eval_config.get("query_strategies", [])
    table_list = eval_config.get("table_extraction", [])

    results: List[dict] = []

    combos = list(itertools.product(chunking_list, retrieval_list, query_list, table_list))
    total_combos = len(combos)
    print(f"\nRunning {total_combos} combination(s)...\n")

    for combo_idx, (chunking, retrieval_mode, q_strategy, table_variant) in enumerate(combos, 1):
        vs_key = f"{chunking}_{table_variant}"
        vector_store = all_vector_stores.get(vs_key)
        if vector_store is None:
            print(
                f"[{combo_idx}/{total_combos}] SKIP — vector store not loaded for key '{vs_key}'"
            )
            continue

        combo_label = f"{chunking} | {retrieval_mode} | {q_strategy} | {table_variant}"
        print(f"\n[{combo_idx}/{total_combos}] Evaluating: {combo_label}")

        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        rag_chain = build_rag_chain(retriever, mode=retrieval_mode)

        tokens_before = cost_tracker.snapshot()
        correct = 0
        total = 0

        for i, sample in enumerate(evaluation_data):
            query = sample["question"]
            expected_answer = sample["original_answer"]
            file_name = sample.get("file_name", "N/A")

            print(f"  [{i+1}/{len(evaluation_data)}] Query: {query}")
            try:
                actual_response = _run_query_with_strategy(rag_chain, query, q_strategy)
            except Exception as exc:
                print(f"  ERROR running query: {exc}")
                actual_response = ""

            comparison = compare_answers(
                query,
                expected_answer,
                actual_response,
                numeric_abs_tol=numeric_abs_tol,
                numeric_rel_tol=numeric_rel_tol,
                judge=judge,
            )
            if comparison.is_correct:
                correct += 1
            else:
                recorder.record(
                    strategy=f"{chunking}_{retrieval_mode}_{q_strategy}_{table_variant}",
                    question=query,
                    expected=expected_answer,
                    actual=actual_response,
                    rationale=comparison.rationale,
                    lexical_context="",
                    semantic_context="",
                    judge_prompt=comparison.judge_prompt,
                    expected_file_name=file_name,
                )
            total += 1
            running_acc = (correct / total) * 100
            print(f"  Running accuracy: {running_acc:.2f}% ({correct}/{total})")

        accuracy = (correct / total * 100) if total > 0 else 0.0
        tokens_after = cost_tracker.snapshot()
        results.append(
            {
                "chunking": chunking,
                "retrieval": retrieval_mode,
                "query_strategy": q_strategy,
                "tables": table_variant,
                "accuracy": f"{accuracy:.2f}%",
                "correct": correct,
                "total": total,
                "llm_calls": tokens_after["calls"] - tokens_before["calls"],
                "input_tokens": tokens_after["input_tokens"] - tokens_before["input_tokens"],
                "output_tokens": tokens_after["output_tokens"] - tokens_before["output_tokens"],
                "total_tokens": tokens_after["total_tokens"] - tokens_before["total_tokens"],
            }
        )

    recorder.save()
    _print_results_table(results)
    _save_results_csv(results)
    cost_tracker.print_summary("Evaluation — LLM Cost Summary")


def _legacy_evaluate(vector_stores: Dict):
    """Original evaluate_strategies behaviour for backwards-compat."""
    evaluation_data = []
    with open("FinQA/test/metadata.jsonl", "r") as f:
        for line in f:
            evaluation_data.append(json.loads(line))

    print(f"Loaded {len(evaluation_data)} evaluation samples.")

    numeric_abs_tol = float(os.getenv("NUMERIC_ABSOLUTE_TOLERANCE", "1e-6"))
    numeric_rel_tol = float(os.getenv("NUMERIC_RELATIVE_TOLERANCE", "0.01"))
    judge = AnswerJudge()
    recorder = WrongAnswerRecorder()

    for name, vector_store in vector_stores.items():
        print(f"\n--- Evaluating {name} strategy ---")
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        rag_chain = build_rag_chain(retriever)
        total_queries = 0
        correct_answers = 0

        for i, sample in enumerate(evaluation_data):
            if i >= 100:
                break
            query = sample["question"]
            expected_answer = sample["original_answer"]
            file_name = sample.get("file_name", "N/A")

            print(f"Query: {query}")
            print(f"File Name: {file_name}")
            response_dict = rag_chain.invoke({"query": query})
            actual_response = response_dict["result"]
            print(f"Expected Answer: {expected_answer}")
            print(f"Actual Response: {actual_response}\n")
            _maybe_print_sources(response_dict)
            lexical_summary, semantic_summary = summarize_by_mode(
                response_dict.get("source_documents") or []
            )
            comparison = compare_answers(
                query,
                expected_answer,
                actual_response,
                numeric_abs_tol=numeric_abs_tol,
                numeric_rel_tol=numeric_rel_tol,
                judge=judge,
            )
            print(f"Is correct: {comparison.is_correct}")
            if comparison.is_correct:
                correct_answers += 1
            else:
                print(f"Marked incorrect: {comparison.rationale}")
                recorder.record(
                    strategy=name,
                    question=query,
                    expected=expected_answer,
                    actual=actual_response,
                    rationale=comparison.rationale,
                    lexical_context=lexical_summary,
                    semantic_context=semantic_summary,
                    judge_prompt=comparison.judge_prompt,
                    expected_file_name=file_name,
                )
            total_queries += 1
            running_accuracy = (correct_answers / total_queries) * 100
            print(f"Running accuracy after {total_queries} samples: {running_accuracy:.2f}%")

        if total_queries > 0:
            accuracy = (correct_answers / total_queries) * 100
            print(f"Accuracy for {name} strategy: {accuracy:.2f}%")
        else:
            print(f"No queries evaluated for {name} strategy.")

    recorder.save()


def _print_results_table(results: List[dict]) -> None:
    if not results:
        print("\nNo results to display.")
        return

    headers = ["Chunking", "Retrieval", "Query Strategy", "Tables", "Correct", "Total", "Accuracy", "LLM Calls", "Input Tok", "Output Tok", "Total Tok"]
    col_keys = ["chunking", "retrieval", "query_strategy", "tables", "correct", "total", "accuracy", "llm_calls", "input_tokens", "output_tokens", "total_tokens"]

    # Compute column widths
    widths = [len(h) for h in headers]
    for row in results:
        for i, key in enumerate(col_keys):
            widths[i] = max(widths[i], len(str(row[key])))

    def fmt_row(values):
        return "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(values)) + " |"

    separator = "|-" + "-|-".join("-" * w for w in widths) + "-|"

    print("\n" + fmt_row(headers))
    print(separator)
    for row in results:
        print(fmt_row([row[k] for k in col_keys]))
    print()


def _save_results_csv(results: List[dict]) -> None:
    if not results:
        return
    os.makedirs("logs", exist_ok=True)
    csv_path = os.path.join("logs", "eval_results.csv")
    fieldnames = ["chunking", "retrieval", "query_strategy", "tables", "correct", "total", "accuracy", "llm_calls", "input_tokens", "output_tokens", "total_tokens"]
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)
    print(f"Results appended to {csv_path}")


@dataclass
class ComparisonResult:
    is_correct: bool
    rationale: str
    judge_prompt: Optional[str] = None


def compare_answers(
    question: str,
    expected: str,
    actual: str,
    *,
    numeric_abs_tol: float,
    numeric_rel_tol: float,
    judge: AnswerJudge,
) -> ComparisonResult:
    expected_value = safe_float(expected)
    actual_value = safe_float(actual)

    if expected_value is not None and actual_value is not None:
        diff = abs(actual_value - expected_value)
        allowed = max(numeric_abs_tol, numeric_rel_tol * max(1.0, abs(expected_value)))
        if diff <= allowed:
            return ComparisonResult(
                True, f"Numeric match within tolerance ({diff:.4g} <= {allowed:.4g})"
            )
        return ComparisonResult(
            False, f"Numeric mismatch ({diff:.4g} > {allowed:.4g})"
        )

    verdict = judge.evaluate(question, expected, actual)
    if verdict is not None:
        decision, rationale = verdict
        return ComparisonResult(decision, rationale, judge_prompt=judge.last_prompt)

    return ComparisonResult(False, "Judge unavailable and no numeric match")


def safe_float(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    multiplier = 1.0
    if text.endswith("%"):
        multiplier = 0.01
        text = text[:-1]
    text = text.replace(",", "")
    try:
        return float(text) * multiplier
    except ValueError:
        return None


def summarize_by_mode(documents: Sequence[Document]):
    lexical_lines = []
    semantic_lines = []
    for idx, doc in enumerate(documents, start=1):
        metadata = doc.metadata or {}
        mode = metadata.get("retrieval_mode", "semantic")
        source = metadata.get("source", "N/A")
        page = metadata.get("page", "N/A")
        score = metadata.get("score", "N/A")
        snippet = (doc.page_content or "").strip().replace("\n", " ")
        snippet = snippet[:500]
        entry = (
            f"{idx}. mode={mode} source={source} page={page} score={score}\n"
            f"{snippet}"
        )
        if mode == "lexical":
            lexical_lines.append(entry)
        else:
            semantic_lines.append(entry)
    return "\n\n".join(lexical_lines), "\n\n".join(semantic_lines)
