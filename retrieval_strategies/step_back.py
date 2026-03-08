from langchain_core.messages import HumanMessage
import re
from utils.llm_factory import build_llm
def query_with_step_back(user_query, model_name=None):
    """
    Decompose a complex user query into logical sub-steps using Chain of Thought prompting
    via Google's Gemini model (using the Generative AI SDK).
    """

    cot_prompt = f"""
You are analyzing financial reports and portfolio data using a RAG model. Your task is to generate "step-back questions" that encourage deeper quantitative analysis of portfolio strategy.

Given a specific financial question, produce step-back questions that:
- Reframe the problem in broader portfolio or macro terms.
- Question assumptions behind the calculations or models.
- Explore risk factors, sensitivities, or alternative scenarios.
- Consider portfolio-level impacts (returns, volatility, drawdowns, diversification).

Focus on questions that could lead to quantitative analysis.

Generate 2 step-back questions.

Format:
[Question 1: ...]
[Question 2: ...]

User query:"{user_query}"
"""

    llm = build_llm(model_name=model_name, temperature=0)
    response = llm.invoke([HumanMessage(content=cot_prompt)])
    # Extract text response
    steps_text = response.content
    # print(f"Response from LLM: {steps_text}")
    extract_questions = lambda text: re.findall(r"\[Question\s*\d+:\s*(.+?)\]", text)
    extract = extract_questions(steps_text)
    # print(f"extracted steps: {extract}")
    return extract


