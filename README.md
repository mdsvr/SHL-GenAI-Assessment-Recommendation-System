# SHL GenAI Assessment Recommendation System

An intelligent assessment recommendation system designed to match hiring requirements with the most suitable SHL assessments using **Semantic Search** and **Retrieval Augmented Generation (RAG)** principles.

## 🚀 Overview

This system leverages vector embeddings and similarity search to bridge the gap between complex job descriptions and a curated catalog of SHL assessments. It provides not only recommendations but also context-aware explanations for why specific assessments were chosen.


*Figure 1: High-level technical architecture of the SHL Assessment Recommendation System.*

The system follows a modular flow, combining **Semantic Search** for retrieval and **RAG-style** processing for recommendation generation and explanation.

### Technical Flow (Developer View)

```mermaid
graph TD
    A[User/Web Browser] -->|Job Requirement| B[Streamlit UI]
    B -->|Query Submission| C{Deployment Mode}
    
    subgraph "Backend Logic (RAG Engine)"
        C -->|Local/Cloud| D[Semantic Encoder]
        D -->|Embedding Vector| E[FAISS Vector Index]
        
        subgraph "Semantic Search"
            E -->|Similarity Retrieval| F[Top-K Assessments]
        end
        
        F -->|Metadata Enrichment| G[Recommendation Engine]
        G -->|Instruction Tuning / Formatting| H[Result Explanation]
    end
    
    H -->|Ranked Recommendations| I[Display Results]
    I -->|Interactive Feedback| A

    classDef highlight fill:#f9f,stroke:#333,stroke-width:2px;
    class E,F highlight;
    
    %% Labels for RAG and Semantic Search
    Note over E,F: **Semantic Search Component**
    Note over D,H: **RAG (Retrieval-Augmented Generation) Flow**
```

### Key Components

*   **Semantic Search**: Uses `SentenceTransformer` (`all-MiniLM-L6-v2`) to encode job descriptions and `FAISS` for lightning-fast vector similarity retrieval.
*   **RAG Flow**: The "Retrieval" of assessment metadata combined with a rule-based "Generation" of explanations ensures that recommendations are both relevant and auditable.
*   **Dual Deployment**: Supports a modular local split (Streamlit + Flask API) and a unified cloud deployment (Single Streamlit App).

## ✨ Features

*   **Intelligent Matching**: Goes beyond keyword search to understand the semantic intent of hiring needs.
*   **Detailed Explanations**: Provides clear rationale for each recommended assessment (Duration, Test Type, Support features).
*   **High Performance**: Uses FAISS for sub-millisecond retrieval even at scale.
*   **Resource Caching**: Optimizes load times by caching the model and vector index.

## 🛠️ Setup & Installation

### Prerequisites
*   Python 3.8+
*   `pip` or `conda`

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd shl-genai-recomender
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃 Running the Application

### 1. Cloud Mode (Single App)
Ideal for quick testing or Streamlit Cloud deployment.
```bash
streamlit run app.py
```

### 2. Local Mode (API + UI)
Ideal for modular development.
*   **Start the API**:
    ```bash
    python api/app.py
    ```
*   **Start the UI**:
    ```bash
    streamlit run ui/app.py
    ```

## 📂 Project Structure
*   `app.py`: Unified Streamlit application (Cloud Mode).
*   `api/`: Flask API implementation for modular deployment.
*   `ui/`: Streamlit frontend that connects to the Flask API.
*   `data/`: Contains the FAISS index (`shl_faiss.index`) and metadata (`shl_metadata.pkl`).
*   `notebooks/`: Research and development notebooks for index creation.

---
*Created as part of the SHL GenAI Assessment Recommendation System Project.*
