# Cybersecurity RAG System

## Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline designed for cybersecurity-focused question answering and threat intelligence assistance. The system combines hybrid document retrieval techniques with Large Language Models (LLMs) to generate context-aware and grounded cybersecurity responses.

The pipeline integrates:

- Dense vector retrieval using Sentence Transformers and FAISS
- Sparse retrieval using BM25
- Hybrid retrieval fusion
- Optional reranking for improved retrieval quality
- Local instruction-tuned LLM generation using Mistral-7B-Instruct
- RAGAS evaluation metrics for automated performance assessment

The system is designed to support cybersecurity tasks such as:

- Vulnerability explanation
- Threat intelligence summarisation
- Malware investigation support
- Detection and mitigation guidance
- Cybersecurity question answering

---

# System Architecture

The pipeline follows the workflow below:

1. Documents are collected and preprocessed.
2. Documents are chunked into smaller text segments.
3. Dense embeddings are generated using Sentence Transformers.
4. Chunks are indexed using FAISS.
5. BM25 sparse retrieval indexes are created.
6. Hybrid retrieval combines semantic and keyword-based search.
7. Retrieved chunks are optionally reranked.
8. Retrieved context is passed to the LLM.
9. The LLM generates grounded cybersecurity responses.
10. RAGAS metrics evaluate retrieval and generation quality.

---

# Project Structure

```bash
RAG/
│
├── data/
│   ├── results/
│   ├── processed/
│   └── test/
│
├── src/
│   ├── chunking.py
│   ├── retrieval.py
│   ├── reranker.py
│   ├── generation.py
│   ├── evaluation.py
│   ├── config.py
│   ├── data_collect.py
│   └── app.py
│
├── requirements.txt
└── README.md
```

---

# Technologies Used

## Core Libraries

- Python
- LangChain
- HuggingFace Transformers
- Sentence Transformers
- FAISS
- BM25
- RAGAS
- Streamlit (still testing)

## Models

### Generator LLM

- Mistral-7B-Instruct-v0.2

### Embedding Model

- sentence-transformers/all-MiniLM-L6-v2

---

# Installation

## Step 1: Clone the repository:

```bash
git clone <repository-url>
cd RAG
```

## Step 2: Install dependencies:

```bash
pip install -r requirements.txt
```



# Running the System

## Step 1: Data Collection

```bash
cd src
python data_collect.py
```

---

## Step 2: Build Embeddings and Indexes

Run preprocessing and indexing scripts:

```bash
python chunking.py
```

---

## Step 3 — Run Retrieval and Generation

```bash
python generation.py
```

---

## Step 4 — Run Evaluation

Make sure you have your test dataset saved under 
    data ->test ->test.csv
```bash
python evaluation.py
```



# Example Query

```text
How is Process Discovery detected
```

Example output:

```text
Process Discovery, also known as Technique T1057, is primarily detected through the
identification of adversarial behaviors related to process ...
```

---


# References

- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
- Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation.
- Sentence Transformers Documentation.
- HuggingFace Transformers Documentation.
- FAISS Similarity Search Documentation.

---

# License

This project is intended for educational and research purposes.

