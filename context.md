# Project Context: Chatbot RAG

This document provides essential context for AI coding tools working on this project. It outlines the project's purpose, key technologies, and architectural overview.

## Project Overview

The Chatbot RAG (Retrieval-Augmented Generation) project implements a conversational AI system capable of answering questions based on provided PDF documents. It integrates various document chunking and retrieval strategies to enhance the accuracy and relevance of the generated responses.

## Key Technologies and Libraries

- **Python**: Primary programming language.
- **LangChain**: Framework for developing applications powered by language models. Used for building the RAG chain, document loading, and text splitting.
- **FAISS**: A library for efficient similarity search and clustering of dense vectors. Used for storing and retrieving document embeddings.
- **Google Generative AI (Gemini)**: Used for generating embeddings and powering the large language model for question answering.
- **uv**: A fast Python package installer and dependency resolver.
- **dotenv**: For loading environment variables from a `.env` file.

## Architectural Overview

The project is structured around the following core components:

1.  **Document Loading and Chunking**: PDF documents are loaded and split into smaller, manageable chunks using various strategies (e.g., Recursive Character, Fixed Size, Sliding Window).
2.  **Vector Store Creation**: Document chunks are embedded using a Google Generative AI embedding model and stored in a FAISS vector database. This allows for efficient semantic search.
3.  **Retrieval-Augmented Generation (RAG) Chain**: When a user asks a question, the system retrieves relevant document chunks from the FAISS vector store. These chunks, along with the user's query, are then fed to a large language model (Gemini) to generate a coherent and informed answer.
4.  **Retrieval Strategies**: The chatbot incorporates advanced retrieval strategies like multi-query generation, chain-of-thought prompting, and step-back decomposition to improve query understanding and response quality.
5.  **Modular Design**: The project is organized into distinct modules for chunking strategies, retrieval strategies, vector store management, and the RAG pipeline, promoting maintainability and extensibility.

## Setup and Execution

Refer to the `README.md` file for detailed instructions on setting up the environment, installing dependencies, creating vector stores, and running the chatbot.
