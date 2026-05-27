# This module prepares the knowledge base for retrieval in
# the RAG pipeline.
# 1. Cleans and preprocesses raw document text.
# 2. Splits documents into smaller overlapping chunks using a sliding window chunking strategy.
# 3. Attaches metadata to every chunk including:
# 4. Generates dense vector embeddings using a SentenceTransformer model.
# 5. Normalizes embeddings for cosine similarity search.
# 6. Saves:

import uuid
import requests
import time
import json
from config import *
import random
from sentence_transformers import SentenceTransformer
import numpy as np
import json, pickle

def clean_text(text):
    if not text:
        return ""
    return " ".join(text.split())

def sliding_window_chunk(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]

        if len(chunk_words) < MIN_CHUNK_WORDS:
            break

        chunks.append(
            " ".join(chunk_words))
        start += chunk_size - overlap
    return chunks

def chunk_document(doc):

    text = clean_text(doc["text"])

    raw_chunks = sliding_window_chunk(text)

    final_chunks = []

    for i, chunk_text in enumerate(raw_chunks):

        chunk = {

            "chunk_id": str(uuid.uuid4()),
            "text": chunk_text,

            # source metadata
            "source": doc.get("source", ""),
            "doc_type": doc.get("doc_type", ""),

            # mitre metadata
            "technique_id": doc.get("technique_id",""),
            "technique_name": doc.get("technique_name",""),
            # mitigation metadata
            "mitigation_id": doc.get("mitigation_id",""),
            "mitigation_name": doc.get("mitigation_name",""),

            # qa metadata
            "question": doc.get("question",""),


            "chunk_index": i,
            "word_count": len(
                chunk_text.split()
            )
        }

        final_chunks.append(chunk)

    return final_chunks

def build_chunks(all_docs):
    all_chunks = []
    for doc in all_docs:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
    return all_chunks

def save_chunks(chunks):
    with open(CHUNKS_PATH,"w",encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

def load_chunks():
    with open(CHUNKS_PATH,"r",encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks")
    return chunks


def create_embeddings(all_chunks, type="float32", transformer="all-MiniLM-L6-v2"):
    embedder = SentenceTransformer(transformer)

    texts = [chunk["text"] for chunk in all_chunks]

    # Generate embeddings 
    print("Generating embeddings...")
    embeddings = embedder.encode(
        texts,
        batch_size=64,        
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = np.array(embeddings).astype(type)

    payload = {
        "model": transformer,
        "chunks": all_chunks,
        "embeddings": embeddings
    }


    with open(EMBEDDINGS_PATH,"wb") as f:
        pickle.dump(payload, f)
    print(f"Saved -{EMBEDDINGS_PATH}")
    return embedder, embeddings


def load_embeddings():
    with open(EMBEDDINGS_PATH, "rb") as f:
        data = pickle.load(f)

    chunks = data["chunks"]
    embeddings = data["embeddings"]

    print(f"Loaded embeddings for {len(chunks)} chunks")

    return chunks, embeddings

if __name__ == "__main__":
    with open(SCRAP, "r") as f:
        all_docs = json.load(f)


    all_chunks = build_chunks(all_docs)

    save_chunks(all_chunks)
    print(f"Total chunks: {len(all_chunks)}")

    embedder, embeddings = create_embeddings(all_chunks)

    print(f"Embeddings shape: {embeddings.shape}")
