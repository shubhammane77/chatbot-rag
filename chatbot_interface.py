from rag_pipeline import build_rag_chain
from retrieval_strategies.multi_query import generate_multi_queries
from retrieval_strategies.chain_of_thought import decompose_query_with_cot
from retrieval_strategies.step_back import query_with_step_back
import json
import os

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
                if "source_documents" in response_dict:
                    if os.getenv("DEBUG_MODE", "false").lower() == "true":
                        print("Retrieved Documents:")
                        for i, doc in enumerate(response_dict["source_documents"]):
                            print(f"  Document {i+1} (Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}):")
                            print(f"    {doc.page_content}\n")
        if query_rewriting.lower() == "c":
            steps = decompose_query_with_cot(query)
            print("\nDecomposed Steps:")
            for i, step in enumerate(steps, 1):
                print(f"{i}. {step}")
            print("\nNow running RAG for each step...")
            for step in steps:
                response_dict = rag_chain.invoke({"query": step})
                print(f"\nAnswer for '{step}':\n", response_dict["result"])
                if "source_documents" in response_dict:
                    if os.getenv("DEBUG_MODE", "false").lower() == "true":
                        print("Retrieved Documents:")
                        for i, doc in enumerate(response_dict["source_documents"]):
                            print(f"  Document {i+1} (Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}):")
                            print(f"    {doc.page_content}\n")
        if query_rewriting.lower() == "s":
            steps = query_with_step_back(query)
            print("\nDecomposed Steps:")
            for i, step in enumerate(steps, 1):
                print(f"{i}. {step}")
            print("\nNow running RAG for each step...")
            for step in steps:
                response_dict = rag_chain.invoke({"query": step})
                print(f"\nAnswer for '{step}':\n", response_dict["result"])
                if "source_documents" in response_dict:
                    if os.getenv("DEBUG_MODE", "false").lower() == "true":
                        print("Retrieved Documents:")
                        for i, doc in enumerate(response_dict["source_documents"]):
                            print(f"  Document {i+1} (Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}):")
                            print(f"    {doc.page_content}\n")
        else:
            response_dict = rag_chain.invoke({"query": query})
            print("\nAnswer:\n", response_dict["result"])
            if "source_documents" in response_dict:
                if os.getenv("DEBUG_MODE", "false").lower() == "true":
                    print("Retrieved Documents:")
                    for i, doc in enumerate(response_dict["source_documents"]):
                        print(f"  Document {i+1} (Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}):")
                        print(f"    {doc.page_content}\n")

def evaluate_strategies(vector_stores):
    evaluation_data = []
    with open("FinQA/dev/metadata.jsonl", "r") as f:
        for line in f:
            evaluation_data.append(json.loads(line))
    
    print(f"Loaded {len(evaluation_data)} evaluation samples.")

    for name, vector_store in vector_stores.items():
        print(f"\n--- Evaluating {name} strategy ---")
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        rag_chain = build_rag_chain(retriever)
        total_queries = 0
        correct_answers = 0
        
        # Limit to a smaller subset for initial testing to avoid excessive API calls
        for i, sample in enumerate(evaluation_data):
            if i >= 10:  # Evaluate only the first 10 samples for now
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
            if "source_documents" in response_dict:
                if os.getenv("DEBUG_MODE", "false").lower() == "true":
                    print("Retrieved Documents:")
                    for i, doc in enumerate(response_dict["source_documents"]):
                        print(f"  Document {i+1} (Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}):")
                        print(f"    {doc.page_content}\n")
            # Simple comparison for demonstration. More sophisticated evaluation would be needed.
            if str(expected_answer).lower() in str(actual_response).lower():
                correct_answers += 1
            total_queries += 1
        
        if total_queries > 0:
            accuracy = (correct_answers / total_queries) * 100
            print(f"Accuracy for {name} strategy: {accuracy:.2f}%")
        else:
            print(f"No queries evaluated for {name} strategy.")
