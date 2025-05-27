from google import genai
from dotenv import load_dotenv, find_dotenv
import os
import logging
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from data import prompt_template
import kagglehub
import pandas as pd

load_dotenv(find_dotenv())
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

COLLECTION_NAME = "sumamrization_base"

client = genai.Client(api_key=os.getenv("API_KEY"))

if __name__ == "__main__":
    qclient = QdrantClient(url="http://localhost:6333")
    try:
        qclient.get_collection(collection_name=COLLECTION_NAME)
    except UnexpectedResponse as e:
        if e.status_code == 404:
            logger.info("Collection not found, creating a new one...")
            qclient.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=384, distance=models.Distance.COSINE
                ),
            )
        else:
            raise
    if qclient.get_collection(collection_name=COLLECTION_NAME).points_count == 0:
        logger.info(
            "Loading dataset from Kaggle: dkhundley/sample-rag-knowledge-item-dataset"
        )
        path_to_save = os.path.join(
            os.getcwd(), "03_summarization", "data", "sample_data.csv"
        )
        if os.path.exists(path_to_save):
            logger.info(
                f"CSV file already exists at {path_to_save}, skipping download."
            )
            csv_files = path_to_save
        else:
            dataset = kagglehub.dataset_download(
                "dkhundley/sample-rag-knowledge-item-dataset"
            )
            # Find the CSV file in the downloaded dataset directory
            csv_files = os.path.join(dataset, "rag_sample_qas_from_kis.csv")
            df = pd.read_csv(csv_files)
            df[["ki_text", "sample_question"]].to_csv(path_to_save, index=False)
        df = pd.read_csv(path_to_save)
        for index, row in df.iterrows():
            text = row["ki_text"]
            contents = [text]
            # Generate embedding using Google GenAI
            try:
                embedding = client.models.embed_content(
                    model="text-embedding-004",
                    contents=contents,
                    config={"output_dimensionality": 384},
                )
                vector = embedding.embeddings[0].values
            except Exception as e:
                logger.error(f"Error generating embedding for text: {text}")
                logger.error(e)
                continue

            qclient.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    models.PointStruct(
                        id=index + 1,
                        vector=vector,
                        payload={"text": text},
                    )
                ],
            )
            logger.info(f"Added text chunk with ID {index} to Qdrant collection.")

    input_query = input("Ask me a question: ")
    transformed_query = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=f"Rephrase the following question for better RAG search results: {input_query}. Do not answer the question, just rephrase it.",
    ).text.strip()
    logger.info(f"Rephrased query: {transformed_query}")

    query_embedding = (
        client.models.embed_content(
            model="text-embedding-004",
            contents=[transformed_query],
            config={"output_dimensionality": 384},
        )
        .embeddings[0]
        .values
    )

    results = qclient.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=3,
    )

    for result in results.points:
        text = result.payload["text"]
        similarity = result.score
        logger.info(f"Text: {text}, Similarity: {similarity:.4f}")

    instruction_prompt = prompt_template.format(
        text_to_summarize="\n".join(
            [
                f"Text: {result.payload['text']}, Similarity: {result.score:.4f}"
                for result in results.points
            ]
        )
    )

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
