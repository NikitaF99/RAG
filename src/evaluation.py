import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from sentence_transformers import SentenceTransformer
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import *
from retrieval import *
from chunking import *
from reranker import *
from generation import *



# LOAD LLM (exisiting generator)
print("Loading offline evaluator LLM...")

generation_pipeline = load_llm()

evaluator_llm = HuggingFacePipeline(
    pipeline=generation_pipeline
)

print("Evaluator LLM loaded.")



# EMBEDDINGS (RAGAS with light weight model)
print("Loading evaluator embeddings...")

evaluator_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Evaluator embeddings loaded.")



# BUILD EVAL DATASET
def build_eval_dataset(path="../data/test/test.csv"):

    print("\nLoading pipeline components...")

    all_chunks, embeddings = load_embeddings()
    index = load_faiss_index()

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    bm25 = load_bm25_index()

    df = pd.read_csv(path)

    rows = []

    for _, row in df.iterrows():

        query = row["question"]
        ground_truth = row["answer"]

        print("\n================================")
        print(f"QUERY: {query}")
        print("================================")

        # =================================================
        # RETRIEVAL
        # =================================================

        retrieved_chunks = retrieve_hybrid(
            query,
            index,
            all_chunks,
            embedder,
            bm25,
            top_k=10
        )

        # =================================================
        # RERANKING
        # =================================================

        retrieved_chunks = rerank(
            query,
            retrieved_chunks,
            top_k=3
        )

        # =================================================
        # GENERATION
        # =================================================

        llm = HuggingFacePipeline(
            pipeline=generation_pipeline
        )

        answer = generate(
            query,
            retrieved_chunks,
            llm
        )

        contexts = [
            " ".join([c["text"] for c in retrieved_chunks])[:1500]
        ]

        print("\nRetrieved Chunks:")

        for i, c in enumerate(retrieved_chunks, 1):
            print(
                f"{i}. "
                f"{c.get('technique_id','')} "
                f"{c.get('doc_type','')}"
            )

        print("\nGenerated Answer:")
        print(answer)

        rows.append({
            "question": query,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth
        })

    return Dataset.from_pandas(pd.DataFrame(rows))



# Save results
def save_results(df):

    detailed_path = "ragas_scores_detailed.csv"
    df.to_csv(detailed_path, index=False)
    print(f"\nSaved detailed results → {detailed_path}")

    summary_cols = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall"
    ]

    available_cols = [c for c in summary_cols if c in df.columns]

    summary_df = df[available_cols]

    summary_path = "ragas_scores_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved summary results → {summary_path}")



def print_summary(df):

    print("\n================================")
    print("RAGAS SUMMARY")
    print("================================")

    metrics = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall"
    ]

    for metric in metrics:
        if metric in df.columns:
            print(f"{metric}: {df[metric].mean():.4f}")



if __name__ == "__main__":

    dataset = build_eval_dataset(TEST_PATH)

    print("\n================================")
    print("RUNNING RAGAS EVALUATION")
    print("================================")

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        raise_exceptions=False
    )

    print("\n================================")
    print("FINAL RESULTS")
    print("================================")

    print(result)

    df = result.to_pandas()

    print_summary(df)

    save_results(df)