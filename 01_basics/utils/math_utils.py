import numpy as np


def cosine_similarity(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def tfidf(term, document, corpus):
    tf = document.count(term) / len(document)

    num_documents_with_term = sum(1 for doc in corpus if term in doc)
    idf = np.log(len(corpus) / (1 + num_documents_with_term))

    return tf * idf
