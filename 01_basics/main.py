import ollama
import logging
from tqdm import tqdm

from utils import load_dataset, cosine_similarity, tfidf

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.CRITICAL)

EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
LANGUAGE_MODEL = "llama3.2"

VECTOR_DB = []


def transform_query(query):
    rephrase = ollama.generate(
        model=LANGUAGE_MODEL,
        prompt=f"Rephrase the following question for better RAG search results: {query}. Do not answer the question, just rephrase it.",
    )
    logger.info(f"Rephrased query: {rephrase['response']}")
    return rephrase["response"].strip()


def add_chunk_to_database(chunk):
    embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)["embeddings"][0]
    VECTOR_DB.append((chunk, embedding))


def retrieve(query, top_n=3):
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)["embeddings"][0]
    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]


def hybrid_search(query, top_n=5):
    """
    Perform a hybrid search using both vector similarity and TF-IDF.
    """
    transformed_query = transform_query(query)
    # Retrieve the top N chunks based on vector similarity
    vector_results = retrieve(transformed_query, top_n * 3)

    tfidf_scores = []
    for chunk, _ in vector_results:
        score = tfidf(transformed_query, chunk, [c[0] for c in VECTOR_DB])
        tfidf_scores.append((chunk, score))

    # Sort by TF-IDF score in descending order
    tfidf_scores.sort(key=lambda x: x[1], reverse=True)

    return tfidf_scores[:top_n]


if __name__ == "__main__":
    dataset = load_dataset("data/cat-facts.txt")

    for i, chunk in tqdm(enumerate(dataset)):
        add_chunk_to_database(chunk)

    input_query = input("Ask me a question: ")
    retrieved_knowledge = hybrid_search(input_query)

    logger.info("Retrieved knowledge:")
    for chunk, similarity in retrieved_knowledge:
        logger.info(f" - (similarity: {similarity:.2f}) {chunk}")

    instruction_prompt = f'''You are a helpful chatbot.
    Use only the following pieces of context to answer the question. Don't make up any new information:
    {'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
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
