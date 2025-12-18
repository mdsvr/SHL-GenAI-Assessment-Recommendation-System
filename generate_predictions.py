import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Load assets
index = faiss.read_index("data/shl_faiss.index")
with open("data/shl_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

# Load test dataset (provided by SHL)
test_df = pd.read_csv("data/train.csv")  
predictions = []

for _, row in test_df.iterrows():
    query = row["Query"]

    q_vec = model.encode([query], convert_to_numpy=True).astype("float32")
    _, idx = index.search(q_vec, 1)  #
    best_match = metadata[idx[0][0]]

    predictions.append({
        "Query": query,
        "Predicted_Assessment_URL": best_match["url"]
    })

# Save predictions
output_file = "vardhan_reddy.csv"  
pd.DataFrame(predictions).to_csv(output_file, index=False)

print(f"Predictions saved to {output_file}")
