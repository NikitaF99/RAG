import faiss
import numpy as np
import pickle
from config import *
from sentence_transformers import SentenceTransformer
import json
from rank_bm25 import BM25Okapi
from chunking import *
from pathlib import Path

FAISS_PATH = "../data/processed/faiss_index.pkl"

def build_faiss_index(embeddings):
    print("Building FAISS index...")

    faiss.normalize_L2(embeddings)

    # Create the index
    dim   = embeddings.shape[1]  
    index = faiss.IndexFlatIP(dim) 

    # Add all embeddings into the index
    index.add(embeddings)

    print(f"Index built — {index.ntotal} vectors stored")

    with open(FAISS_PATH, "wb") as f:
        pickle.dump(index, f)
    print(f"Saved → {FAISS_PATH}")

    return index


def load_faiss_index():
    with open(FAISS_PATH, "rb") as f:
        index = pickle.load(f)
    print(f"Loaded FAISS index with {index.ntotal} vectors")
    return index


def retrieve(query, index, all_chunks, embedder, top_k=3):
    """
    Given a query string, return the top_k most relevant chunks.
    """
    # Embed the query using the same model
    query_emb = embedder.encode([query]).astype("float32")
    faiss.normalize_L2(query_emb)

    # Search the index
    scores, indices = index.search(query_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "text":   all_chunks[idx]["text"],
            "source": all_chunks[idx]["source"],
            "name":   all_chunks[idx]["name"],
            "score":  round(float(score), 4)
        })

    return results




def build_bm25_index(all_chunks):
    print("Building BM25 index...")

    # BM25 works on tokenised text (list of words per chunk)
    tokenised = [chunk["text"].lower().split() for chunk in all_chunks]
    bm25      = BM25Okapi(tokenised)

    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)
    print(f"Saved → {BM25_PATH}")
    return bm25


def load_bm25_index():
    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)
    print("Loaded BM25 index")
    return bm25


def retrieve_dense(query, index, all_chunks, embedder, top_k=3):
    """Pure dense retrieval using FAISS — from previous step."""
    query_emb = embedder.encode([query]).astype("float32")
    faiss.normalize_L2(query_emb)
    scores, indices = index.search(query_emb, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "text":   all_chunks[idx]["text"],
            "source": all_chunks[idx]["source"],
            "name":   all_chunks[idx]["name"],
            "score":  round(float(score), 4)
        })
    return results


def retrieve_hybrid(query, index, all_chunks, embedder, bm25, top_k=3, alpha=0.5):
    """
    Hybrid retrieval — combines dense + BM25 scores.
    
    alpha: weight balance between the two
        1.0 = pure dense
        0.0 = pure BM25
        0.5 = equal mix (default)
    """

    query_emb = embedder.encode([query]).astype("float32")
    faiss.normalize_L2(query_emb)
    # Get more candidates than top_k so we have room to re-rank
    dense_scores, dense_indices = index.search(query_emb, top_k * 10)


    tokenised_query = query.lower().split()
    bm25_scores     = bm25.get_scores(tokenised_query)  # score for every chunk

    # Normalise
    def normalise(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn == 0:
            return arr
        return (arr - mn) / (mx - mn)

    dense_score_arr = np.zeros(len(all_chunks))
    for score, idx in zip(dense_scores[0], dense_indices[0]):
        dense_score_arr[idx] = score

    dense_norm = normalise(dense_score_arr)
    bm25_norm  = normalise(bm25_scores)


    combined = alpha * dense_norm + (1 - alpha) * bm25_norm
    top_indices = np.argsort(combined)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "text":        all_chunks[idx]["text"],
            "source":      all_chunks[idx]["source"],
            "name":        all_chunks[idx]["name"],
            "score":       round(float(combined[idx]), 4),
            "dense_score": round(float(dense_norm[idx]), 4),
            "bm25_score":  round(float(bm25_norm[idx]), 4)
        })
    return results


if __name__ == "__main__":



    all_chunks, embeddings = load_embeddings()


    if Path(FAISS_PATH).exists():
        index = load_faiss_index()
    else:
        with open(EMBEDDINGS_PATH, "rb") as f:
            data = pickle.load(f)
        all_chunks = data["chunks"]
        embeddings  = data["embeddings"]
        index       = build_faiss_index(embeddings)

    embedder               = SentenceTransformer(EMBEDDING_MODEL)
    bm25                   = build_bm25_index(all_chunks)  

   
    queries = [
        "How can we detect phishing attacks?",  # semantic query 
        "T1566.001 spearphishing",               # exact ID query 
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")

        print("\n--- DENSE RESULTS ---")
        dense_results = retrieve_dense(query, index, all_chunks, embedder)
        for i, r in enumerate(dense_results, 1):
            print(f"  {i}. (score: {r['score']}) {r['name']} — {r['text'][:120]}...")

        print("\n--- HYBRID RESULTS ---")
        hybrid_results = retrieve_hybrid(query, index, all_chunks, embedder, bm25)
        for i, r in enumerate(hybrid_results, 1):
            print(f"  {i}. (combined: {r['score']} | dense: {r['dense_score']} | bm25: {r['bm25_score']})")
            print(f"      {r['name']} — {r['text'][:120]}...")