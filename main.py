import getpass
from dotenv import load_dotenv
from load_docs import load_and_split_docs
from multi_query import generate_multi_queries
from vector_store import create_vector_store
from rag_pipeline import build_rag_chain
import os

def main():
    load_dotenv()  # Load environment variables from .env
    os.environ["LANGSMITH_TRACING"] = "true"
    if not os.environ.get("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_API_KEY"] = getpass("Enter your LANGSMITH_API_KEY: ")
    print(f"LANGSMITH_API_KEY: {os.environ.get('LANGSMITH_API_KEY')}")
    os.environ["LANGSMITH_PROJECT"] = "49036947708"
    os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY", "")
    print("Creating vector store...")
    docs = load_and_split_docs()
    vector_store = create_vector_store(docs)


    rag_chain = build_rag_chain(vector_store)

    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        query_rewriting = input("\n Type m for multi-query generation, c for chain of thought prompting: ")
        if query_rewriting.lower() == "m":
            multi_queries = generate_multi_queries(query)
            for mq in multi_queries:
                result = rag_chain.run(mq)
                print(f"\nAnswer for '{mq}':\n", result)
        if query_rewriting.lower() == "c":
            from chainOfThought import decompose_query_with_cot
            steps = decompose_query_with_cot(query)
            print("\nDecomposed Steps:")
            for i, step in enumerate(steps, 1):
                print(f"{i}. {step}")
            print("\nNow running RAG for each step...")
            for step in steps:
                result = rag_chain.run(step)
                print(f"\nAnswer for '{step}':\n", result)        
        else:
            result = rag_chain.run(query)
            print("\nAnswer:\n", result)

if __name__ == "__main__":
    main()
