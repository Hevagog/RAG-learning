from google import genai
from dotenv import load_dotenv, find_dotenv
import os
import logging
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from rank_bm25 import BM25Okapi
from fastembed import TextEmbedding


load_dotenv(find_dotenv())
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

COLLECTION_NAME = "sumamrization_base"
embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

client = genai.Client(api_key=os.getenv("API_KEY"))

if __name__ == "__main__":
    qclient = QdrantClient(url="http://localhost:6333")
    try:
        qclient.get_collection(collection_name=COLLECTION_NAME)
    except UnexpectedResponse as e:
        raise e

    if qclient.get_collection(collection_name=COLLECTION_NAME).points_count != 0:
        input_query = input("Ask me a question: ")
        transformed_query = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=f"Rephrase the following question for better RAG search results: {input_query}. Do not answer the question, just rephrase it.",
        ).text.strip()
        logger.info(f"Rephrased query: {transformed_query}")
        query_embedding = list(embedder.embed(transformed_query))[0]
        bm_25 = BM25Okapi(
            [
                list(embedder.embed(item.payload["text"]))[0]
                for item in qclient.scroll(collection_name=COLLECTION_NAME)[0]
            ]
        )
        results = qclient.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=10,
        )

        bm_scores = bm_25.get_scores(transformed_query.split())
        bm_results = sorted(zip(bm_scores, results), key=lambda x: x[0], reverse=True)[
            :3
        ]
        logger.debug("Hybrid search results:")
        for score, result in bm_results:
            logger.debug(f"Score: {score}, Text: {result}")

        result_texts = [
            f"Text: {result}, Similarity: {score:.4f}" for score, result in bm_results
        ]

        instruction_prompt = f"""You are an expert in helping users find relevant information based on their queries.
            Given the following retrieved text chunks, and their similarity scores, provide a concise and factual answer to the user's question.
            Do not add any information not present in the text. If the information is not present, say "I don't know". Do not make up any information. 
            Retrieved Text with Similarity Scores:
            ---{'\n'.join(result_texts)}
            ---
            User's Question: {input_query}
            Provide a concise answer based on the retrieved text chunks.
        """

        stream = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=[
                instruction_prompt,
                input_query,
            ],
        )
        response = stream.text

        print("Chatbot response:")
        print(response, end="", flush=True)
