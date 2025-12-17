import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------
CSV_PATH = "data/shl_assessments.csv"
FAISS_INDEX_PATH = "data/shl_faiss.index"
METADATA_PATH = "data/shl_metadata.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"
# ----------------------------------------

print("Loading dataset...")
df = pd.read_csv(CSV_PATH)

# Build retrieval text
print("Building retrieval text...")
df["retrieval_text"] = (
    "Assessment Name: " + df["name"].fillna("") + ". "
    "Description: " + df["description"].fillna("") + ". "
    "Test Type: " + df["test_type"].fillna("").astype(str) + ". "
    "Duration: " + df["duration"].fillna(0).astype(int).astype(str) + " minutes."
)

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

# Generate embeddings
print("Generating embeddings...")
embeddings = model.encode(
    df["retrieval_text"].tolist(),
    show_progress_bar=True,
    convert_to_numpy=True
).astype("float32")

# Create FAISS index
print("Creating FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index
faiss.write_index(index, FAISS_INDEX_PATH)

# Save metadata
with open(METADATA_PATH, "wb") as f:
    pickle.dump(df.to_dict(orient="records"), f)

print("\n=== EMBEDDING SUMMARY ===")
print(f"Total assessments indexed : {index.ntotal}")
print(f"Embedding dimension       : {dimension}")
print(f"FAISS index saved to      : {FAISS_INDEX_PATH}")
print(f"Metadata saved to         : {METADATA_PATH}")
print("=========================")
