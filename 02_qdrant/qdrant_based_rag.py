import ollama
import logging
from qdrant_client import QdrantClient


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "qllama/bge-small-en-v1.5:latest"
LANGUAGE_MODEL = "llama3.2"

if __name__ == "__main__":
    client = QdrantClient(url="http://localhost:6333")
    client.get_collection(collection_name=COLLECTION_NAME)
    if client.get_collection(collection_name="knowledge_base").points_count == 0:
        logger.info("Knowledge base is empty, loading dataset...")
        raise NotImplementedError(
            "Please load the dataset into Qdrant before running the script."
        )
        # dataset = load_dataset("src/data/data.json")
        # for chunk in tqdm(dataset, desc="Adding chunks to Qdrant"):
        #     embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)["embeddings"][
        #         0
        #     ]
        #     client.upsert(
        #         collection_name=COLLECTION_NAME,
        #         points=[
        #             {
        #                 "id": chunk["id"],
        #                 "vector": embedding,
        #                 "payload": {"text": chunk["text"]},
        #             }
        #         ],
        #     )
    else:
        input_query = input("Ask me a question: ")
        transformed_query = ollama.generate(
            model=LANGUAGE_MODEL,
            prompt=f"Rephrase the following question for better RAG search results: {input_query}. Do not answer the question, just rephrase it.",
        )["response"].strip()
        logger.info(f"Rephrased query: {transformed_query}")

        query_embedding = ollama.embeddings(
            model=EMBEDDING_MODEL, prompt=transformed_query
        )["embedding"]
        results = client.query_points(
            collection_name=COLLECTION_NAME, query=query_embedding, limit=3
        )

        for result in results.points:
            text = result.payload["document"]
            similarity = result.score
            logger.info(f"Text: {text}, Similarity: {similarity:.4f}")

        instruction_prompt = f'''You are a helpful chatbot.
        Use only the following pieces of context to answer the question. Don't make up any new information:
        {'\n'.join([f"Text: {result.payload['document']}, Similarity: {result.score:.4f}" for result in results.points])}
        '''
        stream = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[
                {"role": "system", "content": instruction_prompt},
                {"role": "user", "content": input_query},
            ],
            stream=True,
        )
        print("Chatbot response:")
        for chunk in stream:
            print(chunk["message"]["content"], end="", flush=True)
        print("\n")
