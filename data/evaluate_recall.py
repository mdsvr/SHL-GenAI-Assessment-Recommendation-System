import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# URL NORMALIZATION (MUST BE DEFINED FIRST)
# --------------------------------------------------
def normalize_url(url):
    if not isinstance(url, str):
        return ""
    url = url.lower().strip()
    url = url.replace("https://www.shl.com", "")
    url = url.replace("/solutions/products/", "")
    url = url.replace("/products/product-catalog/view/", "")
    return url.strip("/")

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
TRAIN_FILE = "data/train.csv"
FAISS_INDEX_PATH = "data/shl_faiss.index"
METADATA_PATH = "data/shl_metadata.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 10

# --------------------------------------------------
# LOAD INDEX & METADATA
# --------------------------------------------------
print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)

print("Loading metadata...")
with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

# --------------------------------------------------
# LOAD TRAIN DATA
# --------------------------------------------------
print("Loading train queries...")
df = pd.read_csv(TRAIN_FILE)

# --------------------------------------------------
# DIAGNOSTIC: CHECK DATASET–INDEX COVERAGE
# --------------------------------------------------
indexed_slugs = set(normalize_url(m["url"]) for m in metadata)

missing = 0
for u in df["Assessment_url"]:
    if normalize_url(u) not in indexed_slugs:
        missing += 1

print(f"\nAssessments NOT in index: {missing} / {len(df)}\n")

# --------------------------------------------------
# RETRIEVAL FUNCTION
# --------------------------------------------------
def retrieve_urls(query, top_k=10):
    query_vec = model.encode([query], convert_to_numpy=True).astype("float32")
    _, indices = index.search(query_vec, top_k)
    return [metadata[i]["url"] for i in indices[0]]

# --------------------------------------------------
# RECALL@10 EVALUATION
# --------------------------------------------------
hits = 0

print("Evaluating Recall@10...\n")

for i, row in df.iterrows():
    query = row["Query"]
    correct_url = row["Assessment_url"]

    retrieved_urls = retrieve_urls(query, TOP_K)

    correct_norm = normalize_url(correct_url)
    retrieved_norms = [normalize_url(u) for u in retrieved_urls]

    hit = int(correct_norm in retrieved_norms)
    hits += hit

    print(f"[{i+1}/{len(df)}] Hit@10 = {hit}")

mean_recall = hits / len(df)

print("\n================ RESULT =================")
print(f"Mean Recall@10: {mean_recall:.4f}")
print("========================================")
