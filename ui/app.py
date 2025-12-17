import streamlit as st
import requests

# ---------------- CONFIG ----------------
API_URL = "http://127.0.0.1:8000/recommend"
# ----------------------------------------

st.set_page_config(
    page_title="SHL Assessment Recommender",
    layout="centered"
)

st.title("SHL GenAI Assessment Recommendation System")
st.write(
    "Paste a job description or hiring query below to get recommended SHL assessments."
)

query = st.text_area(
    "Job Description / Hiring Query",
    height=160,
    placeholder="Hiring a Java developer with strong problem-solving and teamwork skills"
)

top_k = st.slider(
    "Number of recommendations",
    min_value=5,
    max_value=15,
    value=10
)

if st.button("Recommend"):
    if not query.strip():
        st.warning("Please enter a job description or query.")
    else:
        with st.spinner("Fetching recommendations..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"query": query, "top_k": top_k},
                    timeout=60
                )
            except Exception as e:
                st.error(f"Could not connect to API: {e}")
            else:
                if response.status_code != 200:
                    st.error(f"API Error: {response.text}")
                else:
                    data = response.json()
                    recs = data.get("recommendations", [])

                    if not recs:
                        st.info("No recommendations found.")
                    else:
                        st.success(f"Found {len(recs)} recommendations")

                        for i, r in enumerate(recs, 1):
                            st.markdown(f"### {i}. {r.get('assessment_name')}")
                            st.write(f"**URL:** {r.get('assessment_url')}")
                            st.write(f"**Duration:** {r.get('duration')} minutes")
                            st.write(f"**Test Type:** {r.get('test_type')}")
                            st.write(f"**Adaptive Support:** {r.get('adaptive_support')}")
                            st.write(f"**Remote Support:** {r.get('remote_support')}")
                            st.markdown("---")
