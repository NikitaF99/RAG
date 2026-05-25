from sentence_transformers import CrossEncoder


reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Reranking function
def rerank(query, retrieved_chunks, top_k=5):
    pairs = [
        (query, chunk["text"])
        for chunk in retrieved_chunks
    ]

    scores = reranker.predict(pairs)
    reranked = []

    for chunk, score in zip(
        retrieved_chunks,
        scores
    ):

        chunk["rerank_score"] = float(score)
        reranked.append(chunk)

    reranked = sorted(
        reranked,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return reranked[:top_k]