from rag_pipeline import build_rag_chain

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
            from retrieval_strategies.multi_query import generate_multi_queries
            multi_queries = generate_multi_queries(query)
            for mq in multi_queries:
                result = rag_chain.run(mq)
                print(f"\nAnswer for '{mq}':\n", result)
        if query_rewriting.lower() == "c":
            from retrieval_strategies.chain_of_thought import decompose_query_with_cot
            steps = decompose_query_with_cot(query)
            print("\nDecomposed Steps:")
            for i, step in enumerate(steps, 1):
                print(f"{i}. {step}")
            print("\nNow running RAG for each step...")
            for step in steps:
                result = rag_chain.run(step)
                print(f"\nAnswer for '{step}':\n", result) 
        if query_rewriting.lower() == "s":
            from retrieval_strategies.step_back import query_with_step_back
            steps = query_with_step_back(query)
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
