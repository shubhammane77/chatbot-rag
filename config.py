import os
from dotenv import load_dotenv

def load_environment_variables():
    load_dotenv()
    # Set LangSmith environment variables
    os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "true")
    os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
    os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "49036947708")
    os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    # Set Google API key
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY", "")
    os.environ["DOCUMENT_PATH"] = os.getenv("DOCUMENT_PATH", "")

    # Print environment variables for verification (optional)
    print(f"LANGSMITH_API_KEY: {os.environ.get('LANGSMITH_API_KEY')}")
    print(f"LANGSMITH_TRACING: {os.environ.get('LANGSMITH_TRACING')}")
    print(f"LANGSMITH_PROJECT: {os.environ.get('LANGSMITH_PROJECT')}")
    print(f"LANGSMITH_ENDPOINT: {os.environ.get('LANGSMITH_ENDPOINT')}")
    print(f"GOOGLE_API_KEY: {os.environ.get('GOOGLE_API_KEY')}")
    print(f"DOCUMENT_PATH: {os.environ.get('DOCUMENT_PATH')}")
