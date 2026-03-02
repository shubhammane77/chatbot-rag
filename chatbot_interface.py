import json
import os
from dataclasses import dataclass
from typing import Optional, Sequence

from langchain_core.documents import Document

from evaluator.judges import AnswerJudge
from evaluator.wrong_answer_recorder import WrongAnswerRecorder
from logs.logging_utils import is_debug_enabled, log_debug
from retrieval_strategies.retriever import build_rag_chain
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

    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
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

def evaluate_strategies(vector_stores):
    evaluation_data = []
    with open("FinQA/dev/metadata.jsonl", "r") as f:
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
        
        # Limit to a smaller subset for initial testing to avoid excessive API calls
        for i, sample in enumerate(evaluation_data):
            if i >= 100:  # Evaluate only the first 10 samples for now
                break
            query = sample["question"]
            expected_answer = sample["original_answer"]
            file_name = sample.get("file_name", "N/A") # Safely get file_name, default to N/A if not present
            
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
