SCRAP = "../data/mitre_docs.json"
BASE_URL = "https://attack.mitre.org"

# Chunking
CHUNK_SIZE = 350
CHUNK_OVERLAP = 75
MIN_CHUNK_WORDS = 40
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDINGS_PATH = "../data/embeddings.pkl"
CHUNKS_PATH = "../data/chunks.json"
BM25_PATH = "../data/bm25.pkl"

# retrieval
FAISS_PATH = "../data/faiss_index.pkl"

#generation
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


RESULTS_PATH = "../data/results/rag_results.csv"
SCORES_PATH  = "../data/results/ragas_scores.csv"
TEST_PATH = "../data/test/test2.csv"

