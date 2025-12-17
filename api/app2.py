from flask import Flask, request, jsonify, render_template
import faiss, pickle
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

index = faiss.read_index("data/shl_faiss.index")
with open("data/shl_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/recommend", methods=["POST"])
def recommend():
    query = request.form.get("query") or request.json.get("query")
    vec = model.encode([query]).astype("float32")
    _, idxs = index.search(vec, 10)

    results = [{"name": metadata[i]["name"], "url": metadata[i]["url"]} for i in idxs[0]]
    return jsonify(results)

@app.route("/health")
def health():
    return jsonify({"status": "ok"})
