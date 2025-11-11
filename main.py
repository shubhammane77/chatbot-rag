import os
from vectore_store.vector_store import load_vector_store
from chatbot_interface import chat_interface, evaluate_strategies
from config import load_environment_variables

def main():
    load_environment_variables()
    
    # Load vector stores for each strategy
    vector_stores = {}
    for name in ["recursive", "fixed_size", "sliding_window"]:
        index_name = f"faiss_index_{name}"
        if os.path.exists(index_name):
            print(f"Loading vector store for {name} strategy...")
            vector_stores[name] = load_vector_store(index_name=index_name)
        else:
            print(f"Vector store for {name} strategy not found. Please run 'uv run create-vector-store' first.")
            exit()

    while True:
        action = input("\nType 'chat' to start chatbot, 'evaluate' to run evaluations, or 'exit' to quit: ").lower()
        if action == "exit":
            break
        elif action == "chat":
            chat_interface(vector_stores)
        elif action == "evaluate":
            print("\nRunning evaluations...")
            evaluate_strategies(vector_stores)
        else:
            print("Invalid action. Please type 'chat', 'evaluate', or 'exit'.")

if __name__ == "__main__":
    main()
