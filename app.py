import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="SHL Assessment Recommendation System",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 SHL Assessment Recommendation System")
st.caption("Semantic Retrieval using FAISS (Production-Safe)")

# -------------------------------------------------
# Load trained FAISS index and metadata (cached)
# -------------------------------------------------
@st.cache_resource
def load_assets():
    index = faiss.read_index("data/shl_faiss.index")
    with open("data/shl_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, metadata, model

index, metadata, embedder = load_assets()

# -------------------------------------------------
# FAISS retrieval logic
# -------------------------------------------------
def retrieve_assessments(query, top_k):
    query_vec = embedder.encode([query], convert_to_numpy=True).astype("float32")
    _, indices = index.search(query_vec, top_k)

    results = []
    for i in indices[0]:
        row = metadata[i]
        results.append({
            "name": row.get("name", "N/A"),
            "url": row.get("url", ""),
            "duration": row.get("duration", "N/A"),
            "test_type": row.get("test_type", "N/A"),
            "adaptive_support": row.get("adaptive_support", "Unknown"),
            "remote_support": row.get("remote_support", "Unknown"),
        })
    return results

# -------------------------------------------------
# Explanation logic (deterministic & auditable)
# -------------------------------------------------
def generate_explanation(query, assessments):
    explanation = (
        f"Based on the hiring requirement **'{query}'**, the following assessments "
        f"were identified as the most relevant using semantic similarity:\n\n"
    )

    for idx, a in enumerate(assessments, 1):
        explanation += (
            f"**{idx}. {a['name']}**\n"
            f"- Suitable for the required skill set\n"
            f"- Duration: {a['duration']} minutes\n"
            f"- Test Type: {a['test_type']}\n"
            f"- Adaptive Support: {a['adaptive_support']}\n"
            f"- Remote Support: {a['remote_support']}\n\n"
        )

    explanation += (
        "These recommendations were generated using vector similarity search "
        "over SHL’s assessment catalog, ensuring objective and reproducible results."
    )

    return explanation

# -------------------------------------------------
# UI
# -------------------------------------------------
user_query = st.text_area(
    "Describe the role you are hiring for:",
    placeholder="e.g., Java developer with backend and enterprise experience",
    height=120
)

top_k = st.slider("Number of recommendations", min_value=3, max_value=15, value=5)

if st.button("Get Recommendations"):
    if not user_query.strip():
        st.warning("Please enter a valid job description.")
        st.stop()

    with st.spinner("Finding best matching assessments..."):
        recommendations = retrieve_assessments(user_query, top_k)

    st.subheader("🔎 Recommended Assessments")

    for r in recommendations:
        st.markdown(
            f"""
**{r['name']}**  
- Duration: {r['duration']} mins  
- Test Type: {r['test_type']}  
- Adaptive: {r['adaptive_support']}  
- Remote: {r['remote_support']}  
- URL: {r['url']}
"""
        )

    st.subheader("🧠 Recommendation Explanation")
    explanation = generate_explanation(user_query, recommendations)
    st.write(explanation)
