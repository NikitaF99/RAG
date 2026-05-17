import os
from transformers import pipeline
from retrieval import *
from chunking import *


LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
HF_CACHE = "/Volumes/NikitaPen/huggingface"

def load_llm():
    print(f"Loading LLM: {LLM_MODEL}")
    print("This may take a few minutes on first run...")
    llm = pipeline(
        "text-generation",
        model=LLM_MODEL,
        device_map="auto",       # uses GPU if available, else CPU
        torch_dtype="auto"
    )
    print("LLM loaded.")
    return llm


def build_prompt(query, retrieved_chunks):
    """
    Builds the RAG prompt — context + query combined.
    """
    # Join the 3 chunks into one context block
    context = "\n\n---\n\n".join([
        f"Source: {c['source']}\n{c['text']}"
        for c in retrieved_chunks
    ])

    prompt = f"""[INST] You are a cybersecurity advisor helping security practitioners.
Use ONLY the context provided below to answer the question.
Do not add any information that is not in the context.

CONTEXT:
{context}

QUESTION:
{query}

Provide exactly 3 concise, actionable security guidelines based on the context above.
Format your response as a numbered list. [/INST]"""

    return prompt


def generate(query, retrieved_chunks, llm):
    """
    Full RAG generation — takes a query + retrieved chunks,
    returns the generated guidelines.
    """
    prompt = build_prompt(query, retrieved_chunks)

    output = llm(
        prompt,
        max_new_tokens=400,
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
    chunks  = retrieve_hybrid(query, index, all_chunks, embedder, bm25, top_k=3)

    # Step 2 — generate
    answer  = generate(query, chunks, llm)

    # Step 3 — print everything
    print(f"\nQUERY:\n{query}")
    print(f"\nRETRIEVED CHUNKS:")
    for i, c in enumerate(chunks, 1):
        print(f"  {i}. {c['name']} (score: {c['score']})")
        print(f"     {c['text'][:150]}...")

    print(f"\nGENERATED ANSWER:\n{answer}")