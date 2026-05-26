import streamlit as st
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from transformers import pipeline
from pathlib import Path
from retrieval import *
from chunking import *
from generation import *
from config import *

# THIS CODE IS STILL UNDER DEVELOPMENT - AND NOT PROPERLY TESTED - NOT SURE OF USING IT

# LOAD EVERYTHING ONCE (cached )
@st.cache_resource
def load_all():
    all_chunks, embeddings = load_embeddings()
    index                  = load_faiss_index()
    embedder               = SentenceTransformer(EMBEDDING_MODEL)
    bm25                   = load_bm25_index()
    llm                    = pipeline(
                                "text-generation",
                                model=LLM_MODEL,
                                device_map="auto",
                                torch_dtype="auto"
                             )
    return all_chunks, index, embedder, bm25, llm

# UI
st.title("CyberRAG")
st.caption("Ask anything about MITRE ATT&CK techniques, mitigations, and detections.")

all_chunks, index, embedder, bm25, llm = load_all()

# Keep chat history across messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if query := st.chat_input("e.g. How do I detect lateral movement?"):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Retrieve and generate answer
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            chunks = retrieve_hybrid(query, index, all_chunks, embedder, bm25, top_k=3)
            answer = generate(query, chunks, llm)

        st.markdown(answer)

        # Show retrieved sources in an expander
        with st.expander("View retrieved sources"):
            for i, c in enumerate(chunks, 1):
                st.markdown(f"**{i}. {c['name']}** (score: {c['score']})")
                st.caption(c["source"])
                st.text(c["text"][:300] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})