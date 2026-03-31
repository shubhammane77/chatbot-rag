"""Retrieval-only precision evaluation.

Runs the retriever (FAISS / ES / hybrid) for each evaluation sample
WITHOUT calling the LLM.  For every retrieved document it checks whether
the source file matches the expected FinQA ``file_name``, then aggregates:

  - precision per query  = relevant_docs / total_retrieved_docs
  - average precision across all queries per (chunking, retrieval_mode, table_variant)

Usage:
    uv run eval-retrieval-precision
"""

import csv
import itertools
import json
import os
from typing import List, Optional

from config import load_environment_variables, load_eval_config
from retrieval_strategies.retriever import create_hybrid_retriever
from ingestion.vector_store.vector_store import load_vector_store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_file_name(file_name: str) -> str:
    """Strip leading 'pdf/' so the key matches doc source paths."""
    return file_name.split("pdf/", 1)[-1]


def _retrieve(retriever, query: str):
    try:
        return retriever.invoke(query)
    except Exception as exc:
        print(f"    WARN: retrieval failed — {exc}")
        return []


def _run_combo(
    vector_store,
    retrieval_mode: str,
    evaluation_data: List[dict],
    sample_count: int,
) -> dict:
    """Run retrieval-only precision for one (vector_store, retrieval_mode) combo."""
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    retriever = create_hybrid_retriever(base_retriever, mode=retrieval_mode)

    precision_sum = 0.0
    precision_count = 0
    total_retrieved = 0
    total_relevant_docs = 0  # actual count of relevant docs across all queries
    total_hits = 0

    for i, sample in enumerate(evaluation_data[:sample_count]):
        query = sample["question"]
        file_name = sample.get("file_name", "")

        docs = _retrieve(retriever, query)
        total_retrieved += len(docs)

        if docs and file_name:
            norm_expected = _norm_file_name(file_name)
            relevant_count = sum(
                1 for doc in docs
                if norm_expected in (doc.metadata.get("source") or "")
            )
            total_relevant_docs += relevant_count
            hit = 1 if relevant_count > 0 else 0
            total_hits += hit
            precision = relevant_count / len(docs)
            precision_sum += precision
            precision_count += 1
            print(
                f"  [{i+1}/{sample_count}] precision={precision:.0%} "
                f"relevant={relevant_count}/{len(docs)} hit={hit} — {file_name}"
            )
        else:
            print(f"  [{i+1}/{sample_count}] no docs retrieved — {file_name}")

    avg_precision = (precision_sum / precision_count * 100) if precision_count > 0 else 0.0
    overall_precision = (total_relevant_docs / total_retrieved * 100) if total_retrieved > 0 else 0.0
    hit_rate = (total_hits / precision_count * 100) if precision_count > 0 else 0.0

    return {
        "avg_query_precision": avg_precision,
        "overall_precision": overall_precision,  # total_relevant_docs / total_retrieved
        "hit_rate": hit_rate,
        "total_retrieved": total_retrieved,
        "total_relevant_docs": total_relevant_docs,
        "hits": total_hits,
        "evaluated": precision_count,
    }


# ---------------------------------------------------------------------------
# Display / save
# ---------------------------------------------------------------------------

def _print_table(results: List[dict]) -> None:
    if not results:
        print("\nNo results.")
        return

    headers = [
        "Chunking", "Retrieval", "Tables",
        "Avg Query Prec.", "Overall Prec.", "Hit Rate",
        "Total Retrieved", "Total Relevant", "Hits", "Samples",
    ]
    col_keys = [
        "chunking", "retrieval", "tables",
        "avg_query_precision", "overall_precision", "hit_rate",
        "total_retrieved", "total_relevant_docs", "hits", "evaluated",
    ]

    widths = [len(h) for h in headers]
    for row in results:
        for i, key in enumerate(col_keys):
            widths[i] = max(widths[i], len(str(row[key])))

    def fmt(values):
        return "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(values)) + " |"

    sep = "|-" + "-|-".join("-" * w for w in widths) + "-|"
    print("\n" + fmt(headers))
    print(sep)
    for row in results:
        print(fmt([row[k] for k in col_keys]))
    print()


def _save_csv(results: List[dict]) -> None:
    if not results:
        return
    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", "retrieval_precision.csv")
    fieldnames = [
        "chunking", "retrieval", "tables",
        "avg_query_precision", "overall_precision", "hit_rate",
        "total_retrieved", "total_relevant_docs", "hits", "evaluated",
    ]
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)
    print(f"Results appended to {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    load_environment_variables()
    eval_config = load_eval_config()

    sample_count = eval_config.get("sample_count", 100)
    chunking_list = eval_config.get("chunking_strategies", [])
    retrieval_list = eval_config.get("retrieval_modes", [])
    table_list = eval_config.get("table_extraction", [])

    evaluation_data: List[dict] = []
    with open("FinQA/test/metadata.jsonl", "r") as f:
        for line in f:
            evaluation_data.append(json.loads(line))
    print(f"Loaded {len(evaluation_data)} samples (using first {sample_count}).")

    combos = list(itertools.product(chunking_list, retrieval_list, table_list))
    print(f"Running {len(combos)} combo(s) — retrieval only, no LLM calls.\n")

    results = []

    for combo_idx, (chunking, retrieval_mode, table_variant) in enumerate(combos, 1):
        if table_variant == "with_tables":
            index_name = f"faiss_indices/{chunking}"
        else:
            index_name = f"faiss_indices/{chunking}_no_tables"

        if not os.path.exists(index_name):
            print(f"[{combo_idx}/{len(combos)}] SKIP — '{index_name}' not found")
            continue

        combo_label = f"{chunking} | {retrieval_mode} | {table_variant}"
        print(f"[{combo_idx}/{len(combos)}] {combo_label}")

        vs = load_vector_store(index_name=index_name)
        stats = _run_combo(vs, retrieval_mode, evaluation_data, sample_count)

        results.append(
            {
                "chunking": chunking,
                "retrieval": retrieval_mode,
                "tables": table_variant,
                "avg_query_precision": f"{stats['avg_query_precision']:.2f}%",
                "overall_precision": f"{stats['overall_precision']:.2f}%",
                "hit_rate": f"{stats['hit_rate']:.2f}%",
                "total_retrieved": stats["total_retrieved"],
                "total_relevant_docs": stats["total_relevant_docs"],
                "hits": stats["hits"],
                "evaluated": stats["evaluated"],
            }
        )

    _print_table(results)
    _save_csv(results)


if __name__ == "__main__":
    main()
