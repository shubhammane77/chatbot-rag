import os
from ingestion.vector_store.vector_store import load_vector_store
from chatbot_interface import chat_interface, evaluate_strategies
from config import load_environment_variables, load_eval_config


def main():
    load_environment_variables()
    eval_config = load_eval_config()

    # Load all required vector store variants: chunking × table_extraction
    # Key format: "{chunking_strategy}_{table_variant}"
    # Index name format: "faiss_index_{strategy}" (with_tables) or "faiss_index_{strategy}_no_tables"
    all_vector_stores = {}
    chat_vector_stores = {}  # simple {name: vs} for chat mode (with_tables only)

    for chunking in eval_config["chunking_strategies"]:
        for table_variant in eval_config["table_extraction"]:
            if table_variant == "with_tables":
                index_name = f"faiss_indices/{chunking}"
            else:
                index_name = f"faiss_indices/{chunking}_no_tables"

            vs_key = f"{chunking}_{table_variant}"
            if os.path.exists(index_name):
                print(f"Loading vector store '{index_name}'...")
                vs = load_vector_store(index_name=index_name)
                all_vector_stores[vs_key] = vs
                if table_variant == "with_tables":
                    chat_vector_stores[chunking] = vs
            else:
                print(
                    f"Warning: vector store '{index_name}' not found — "
                    f"skipping combination '{vs_key}'. "
                    f"Run 'uv run create-vector-store' to build it."
                )

    if not all_vector_stores:
        print(
            "No vector stores could be loaded. "
            "Please run 'uv run create-vector-store' first."
        )
        return

    # Fall back: use first available variant per chunking strategy for chat
    if not chat_vector_stores:
        for chunking in eval_config["chunking_strategies"]:
            for table_variant in eval_config["table_extraction"]:
                vs_key = f"{chunking}_{table_variant}"
                if vs_key in all_vector_stores:
                    chat_vector_stores[chunking] = all_vector_stores[vs_key]
                    break

    while True:
        action = input(
            "\nType 'chat' to start chatbot, 'evaluate' to run evaluations, or 'exit' to quit: "
        ).lower()
        if action == "exit":
            break
        elif action == "chat":
            chat_interface(chat_vector_stores)
        elif action == "evaluate":
            print("\nRunning evaluations...")
            evaluate_strategies(all_vector_stores, eval_config=eval_config)
        else:
            print("Invalid action. Please type 'chat', 'evaluate', or 'exit'.")


if __name__ == "__main__":
    main()
