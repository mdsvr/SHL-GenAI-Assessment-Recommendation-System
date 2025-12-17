from flask import Flask, request, jsonify
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------
FAISS_INDEX_PATH = "data/shl_faiss.index"
METADATA_PATH = "data/shl_metadata.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K_DEFAULT = 10
# ----------------------------------------

app = Flask(__name__)

# Load resources once
index = faiss.read_index(FAISS_INDEX_PATH)
with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

model = SentenceTransformer(MODEL_NAME)

def normalize_url(url):
    if not isinstance(url, str):
        return ""
    url = url.lower().strip()
    url = url.replace("https://www.shl.com", "")
    url = url.replace("/solutions/products/", "")
    url = url.replace("/products/product-catalog/view/", "")
    return url.strip("/")

def retrieve(query, top_k):
    q_vec = model.encode([query], convert_to_numpy=True).astype("float32")
    _, idxs = index.search(q_vec, top_k)
    results = []
    for i in idxs[0]:
        row = metadata[i]
        results.append({
            "assessment_name": row.get("name"),
            "assessment_url": row.get("url"),
            "duration": int(row.get("duration", 0)),
            "test_type": row.get("test_type"),
            "adaptive_support": row.get("adaptive_support"),
            "remote_support": row.get("remote_support")
        })
    return results

@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "name": "SHL Assessment Recommender API",
        "version": "1.0",
        "endpoints": {
            "GET /health": "Health check",
            "POST /recommend": "Get assessment recommendations (requires 'query' in JSON body)"
        }
    }), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json(force=True, silent=True) or {}
    query = data.get("query", "").strip()
    top_k = int(data.get("top_k", TOP_K_DEFAULT))

    if not query:
        return jsonify({"error": "query is required"}), 400

    recs = retrieve(query, top_k)
    return jsonify({
        "query": query,
        "top_k": top_k,
        "recommendations": recs
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
