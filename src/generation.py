import os
from transformers import pipeline
from retrieval import *
from chunking import *
from config import *



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


def build_prompt(query, retrieved_chunks, mode="production"):

    context_blocks = []
    prompt = ""
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

    if mode == "production":
        prompt = f"""[INST]
        You are a cybersecurity assistant.

        Use ONLY the provided context.

        Do NOT add extra formatting or labels like "Explanation:" or "Detection Guidance:".

        Answer in a natural, technical paragraph style.

        If relevant, include:
        - detection ideas
        - mitigation ideas

        Use inline citations like [Source 1], [Source 2].

        CONTEXT:
        {context}

        QUESTION:
        {query}
        [/INST]"""
    elif mode == "eval":

        prompt = f"""[INST]
    You are a cybersecurity expert.

    Use ONLY the provided context.

    Answer the question in a clear, concise paragraph.

    Do NOT use headings, sections, or bullet formatting.

    CONTEXT:
    {context}

    QUESTION:
    {query}
    [/INST]"""
    return prompt


def generate(query, retrieved_chunks, llm, mode="production"):
    """
    Full RAG generation — takes a query + retrieved chunks,
    returns the generated guidelines.
    """
    prompt = build_prompt(query, retrieved_chunks, mode=mode)

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
    chunks  = retrieve_hybrid(query, index, all_chunks, embedder, bm25, top_k=3)

    # Step 2 — generate
    answer  = generate(query, chunks, llm)

    # Step 3 — print everything
    print(f"\nQUERY:\n{query}")
    print(f"\nRETRIEVED CHUNKS:")
    for i, c in enumerate(chunks, 1):

        print(f"\n{i}. [{c.get('doc_type')}] {c.get('name')}")

        print(f"Score: {c['score']}")

        print(f"{c['text'][:150]}...")


    print(f"\nGENERATED ANSWER:\n{answer}")