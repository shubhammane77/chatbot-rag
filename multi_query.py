from typing import List
import random

def lexical_reformulation(question: str) -> List[str]:
    return [
        question.replace("How does", "What is the process by which"),
        question.replace("How", "In what way"),
        question.replace("work", "function"),
    ]

def semantic_expansion(question: str) -> List[str]:
    return [
        f"What are some real-world applications of {question.split(' in ')[-1]}?",
        f"What concepts are related to {question.lower()}?",
        f"How is {question.split()[-1]} used in different industries?",
    ]

def perspective_shift(question: str) -> List[str]:
    return [
        f"As an engineer, how would you implement {question.lower()}?",
        f"What do researchers say about {question.lower()}?",
        f"How does industry apply {question.lower()} in real projects?",
    ]

def temporal_framing(question: str) -> List[str]:
    return [
        f"How has {question.lower()} evolved over the years?",
        f"What are the current trends in {question.lower()}?",
        f"What future advancements are expected in {question.lower()}?",
    ]

def comparative_thinking(question: str) -> List[str]:
    return [
        f"How does {question.lower()} compare to supervised learning?"    ]

def failure_or_limitation(question: str) -> List[str]:
    return [
        f"What are the limitations of {question.lower()}?",
        f"Why might {question.lower()} not work in all cases?",
    ]

def generate_multi_queries(question: str) -> List[str]:
    queries = []
    queries.extend(lexical_reformulation(question))
    queries.extend(semantic_expansion(question))
    queries.extend(perspective_shift(question))
    queries.extend(temporal_framing(question))
    queries.extend(comparative_thinking(question))
    queries.extend(failure_or_limitation(question))
    
    # Remove duplicates and clean up
    queries = list(set([q.strip() for q in queries if q.strip() != ""]))
    return queries
