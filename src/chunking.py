import requests
import time
import json
from config import *
import random
from sentence_transformers import SentenceTransformer
import numpy as np
import json, pickle

def chunk_text(doc, chunk_size=400, overlap=50):
    """
    Splits a document's text into overlapping chunks.
    """
    words = doc["text"].split()
    chunks = []
    
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        
   
        if len(chunk_words) > 30:
            chunks.append({
                "text": chunk_text,
                "source": doc["source"],
                "technique_id": doc.get("technique_id", ""),
                "name": doc.get("name", "")
            })
        
        start += chunk_size - overlap  # slide window with overlap
    
    return chunks

def create_embeddings(all_chunks, type="float32", transformer="all-MiniLM-L6-v2"):
    embedder = SentenceTransformer(transformer)

    texts = [chunk["text"] for chunk in all_chunks]

    # Generate embeddings 
    print("Generating embeddings...")
    embeddings = embedder.encode(
        texts,
        batch_size=64,        
        show_progress_bar=True
    )

    embeddings = np.array(embeddings).astype(type)

    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump({"chunks": all_chunks, "embeddings": embeddings}, f)
    print(f"Saved → {EMBEDDINGS_PATH}")
    return embedder, embeddings, all_chunks

def save_chunks(chunks):
    """
    Save chunks to JSON.
    """

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"Saved chunks → {CHUNKS_PATH}")


def load_embeddings():
    with open(EMBEDDINGS_PATH, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded embeddings for {len(data['chunks'])} chunks")
    return data["chunks"], data["embeddings"]

if __name__ == "__main__":
    with open(SCRAP, "r") as f:
        all_docs = json.load(f)


    all_chunks = []
    for doc in all_docs:
        all_chunks.extend(chunk_text(doc))

    save_chunks(all_chunks)
    print(f"Total chunks: {len(all_chunks)}")

    embedder, embeddings, all_chunks = create_embeddings(all_chunks)

    # all_docs              = load_knowledge_base()
    # all_chunks, embeddings = load_embeddings()
    print(f"Embeddings shape: {embeddings.shape}")
