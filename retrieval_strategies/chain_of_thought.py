from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
import re
def decompose_query_with_cot(user_query, model_name="gemini-2.5-flash"):
    """
    Decompose a complex user query into logical sub-steps using Chain of Thought prompting
    via Google's Gemini model (using the Generative AI SDK).
    """

    cot_prompt = f"""
You are an AI assistant helping to prepare a scientific retrieval task.

Your goal is to analyze the user's research question and break it down into exactly **3 logical reasoning steps** or **sub-questions**. These steps should be focused and useful for retrieving relevant information from scientific research papers.

Please follow this response format:
[Step 1: ...]
[Step 2: ...]
[Step 3: ...]

User query: "{user_query}"

Think step by step and return only the list.
"""

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    response = llm([HumanMessage(content=cot_prompt)])
    # Extract text response
    steps_text = response.content
    # print(f"Response from LLM: {steps_text}")
    extract_questions = lambda text: re.findall(r"\[Step\s*\d+:\s*(.+?)\]", text)
    extract = extract_questions(steps_text)
    # print(f"extracted steps: {extract}")
    return extract


