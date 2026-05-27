# This module implements the generation component of the RAG pipeline.
# 1. Loads the fine-tuned LLM using HuggingFace Transformers.
# 2.Retrieves relevant chunks from the knowledge base using the retrieval module.
#  3. Reranks retrieved chunks using a cross-encoder for better relevance.
# 4. Builds a prompt template that incorporates retrieved chunks as context.
# 5. Generates an answer using the LLM based on the prompt.

import os
from transformers import pipeline
from retrieval import *
from chunking import *
from config import *
from reranker import *


def load_llm():
    print(f"Loading LLM: {LLM_MODEL}")
    llm = pipeline(
        "text-generation",
        model=LLM_MODEL,
        device_map="auto",      
        torch_dtype="auto"
    )
    print("LLM loaded.")
    return llm

# Prompt engineering
def build_prompt(query, retrieved_chunks):

    context_blocks = []
    for i, c in enumerate(retrieved_chunks, 1):

        block = f"""
            [Source {i}]
            Type: {c.get('doc_type', 'unknown')}
            Technique: {c.get('technique_id', '')} {c.get('name', '')}
            Content:
            {c['text']}
            """
        context_blocks.append(block)
    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""
        You are a professional cybersecurity assistant.

        Use ONLY the provided context.

        If the context is insufficient, say:
        "I could not find sufficient information."

        Answer in a clear, concise paragraph. No bullets or numbering

        Requirements:
                - concise
                - actionable
                - cybersecurity-focused
                - professional
                - no hallucinations
                - cite sources whether MITRE or QA when relevant


        CONTEXT:
        {context}

        QUESTION:
        {query}

        ANSWER:
        """

    return prompt


def generate(query, retrieved_chunks, llm):
    """
    Full RAG generation — takes a query + retrieved chunks,
    returns the generated guidelines.
    """
    prompt = build_prompt(query, retrieved_chunks)

    output = llm(
        prompt,
        max_new_tokens=500,
        do_sample=False,       
        repetition_penalty=1.1 # avoids the model repeating itself
    )

    # Extract only the generated
    full_text = output[0]["generated_text"]
    answer    = full_text.split("[/INST]")[-1].strip()
    return answer

if __name__ == "__main__":

    # Load everything
    all_chunks, embeddings = load_embeddings()
    index                  = load_faiss_index()
    embedder               = SentenceTransformer(EMBEDDING_MODEL)
    bm25                   = load_bm25_index()
    llm                    = load_llm()

    query   = "How should we handle spear-phishing attempts in a mid-sized enterprise?"

    # Step 1 — retrieve
    chunks  = retrieve_hybrid(query, index, all_chunks, embedder, bm25, top_k=10)
    # Step 2 - reranking step
    chunks = rerank(
        query,
        chunks,
        top_k=3
    )
    # Step 3 — generate
    answer  = generate(query, chunks, llm)

    # Step 4 — print everything
    print(f"\nQUERY:\n{query}")
    print(f"\nRETRIEVED CHUNKS:")
    for i, c in enumerate(chunks, 1):

        print(f"\n{i}. [{c.get('doc_type')}] {c.get('name')}")

        print(f"Score: {c['score']}")

        print(f"{c['text'][:150]}...")


    print(f"\nGENERATED ANSWER:\n{answer}")