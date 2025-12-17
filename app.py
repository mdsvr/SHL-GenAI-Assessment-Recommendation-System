import gradio as gr
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------
FAISS_INDEX_PATH = "data/shl_faiss.index"
METADATA_PATH = "data/shl_metadata.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K_DEFAULT = 10
# ----------------------------------------

print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)

print("Loading metadata...")
with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

def retrieve_assessments(query, top_k):
    if not query.strip():
        return "Please enter a valid job description."

    query_vec = model.encode([query], convert_to_numpy=True).astype("float32")
    _, indices = index.search(query_vec, top_k)

    results = []
    for i in indices[0]:
        row = metadata[i]
        results.append(
            f"""
### {row.get('name', 'Unknown')}
- **URL:** {row.get('url', '')}
- **Duration:** {row.get('duration', 'Unknown')} minutes
- **Test Type:** {row.get('test_type', 'Unknown')}
- **Adaptive Support:** {row.get('adaptive_support', 'Unknown')}
- **Remote Support:** {row.get('remote_support', 'Unknown')}
"""
        )

    return "\n".join(results)

with gr.Blocks(title="SHL GenAI Assessment Recommendation") as demo:
    gr.Markdown("# SHL GenAI Assessment Recommendation System")
    gr.Markdown(
        "Enter a job description or hiring requirement to get recommended SHL assessments."
    )

    query_input = gr.Textbox(
        label="Job Description / Hiring Query",
        lines=6,
        placeholder="Hiring a customer service executive with sales and communication skills"
    )

    top_k_input = gr.Slider(
        minimum=5,
        maximum=15,
        value=TOP_K_DEFAULT,
        step=1,
        label="Number of Recommendations"
    )

    output = gr.Markdown()

    submit_btn = gr.Button("Recommend")

    submit_btn.click(
        retrieve_assessments,
        inputs=[query_input, top_k_input],
        outputs=output
    )

demo.launch()
