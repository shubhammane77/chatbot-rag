from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
import re
def query_with_step_back(user_query, model_name="gemini-2.5-flash"):
    """
    Decompose a complex user query into logical sub-steps using Chain of Thought prompting
    via Google's Gemini model (using the Generative AI SDK).
    """

    cot_prompt = f"""
You are analyzing a complex research paper using a RAG model. Your goal is to generate "stepback questions" — high-level, reflective questions that encourage deeper understanding, challenge assumptions, or explore the broader context of the research.

Given a specific question about the research paper, produce several stepback questions that:
- Re-express the problem in broader or more general terms.
- Question the assumptions underlying the research.
- Connect the topic to foundational theories or related fields.
- Consider the implications or consequences of the findings.
- Explore alternative perspectives or formulations.

Example Input Question: "What are the key findings of this paper on transformer-based models for protein folding?"

Example Stepback Questions:
- What assumptions does this paper make about the relationship between language modeling and protein structure?
- How do transformer-based models compare with traditional approaches in terms of interpretability?
- What limitations in the training data might affect generalization to novel proteins?
- What does success in this domain suggest about the generality of transformer architectures?

Now, for the following input question, generate 3 stepback questions:
Please follow this response format:
[Question 1: ...]
[Question 2: ...]
[Question 3: ...]
[Insert Input Question Here]
User query: "{user_query}"
"""

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    response = llm([HumanMessage(content=cot_prompt)])
    # Extract text response
    steps_text = response.content
    # print(f"Response from LLM: {steps_text}")
    extract_questions = lambda text: re.findall(r"\[Question\s*\d+:\s*(.+?)\]", text)
    extract = extract_questions(steps_text)
    # print(f"extracted steps: {extract}")
    return extract


