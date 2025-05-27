# simple-rag

This repository is a collection of small projects and experiments focused on learning and exploring Retrieval Augmented Generation (RAG) techniques and Large Language Models (LLMs).

## Projects

### 01_basics
This project serves as an introduction to the core concepts of RAG. It implements a basic RAG pipeline from scratch, demonstrating how to:
*   Load and process a text dataset (e.g., `cat-facts.txt`).
*   Generate embeddings for text chunks.
*   Perform similarity search to retrieve relevant context.
*   Combine the retrieved context with a user query to generate an answer using an LLM.

### 02_qdrant
This project explores using Qdrant, a vector database, to build a more robust RAG system. Key aspects include:
*   Setting up and interacting with a Qdrant instance.
*   Creating collections and defining vector parameters.
*   Upserting documents and their embeddings into Qdrant.
*   Querying Qdrant to find relevant documents based on semantic similarity.
*   Integrating Qdrant-retrieved context into an LLM-powered question-answering pipeline.
*   The `qtest.ipynb` notebook provides a hands-on way to experiment with Qdrant operations and populate the database using data from `data.json`.
