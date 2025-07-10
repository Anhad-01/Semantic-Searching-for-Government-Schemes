# Semantic Search Dashboard for Government Schemes

This project provides a semantic search dashboard for government schemes using Streamlit, FAISS, and Sentence Transformers.

## Features
- Semantic search over government schemes using multilingual sentence embeddings
- **Side-by-side comparison of Euclidean (L2) and Cosine Similarity**: See how different similarity metrics affect your search results
- Smart filtering by Indian states and central (ministry) schemes
- **Explicit state filter**: Select a state from the sidebar to filter results, or let the system detect the state from your query
- **User-friendly search controls**: Prominent search input, primary search button, and a clear search button to reset your query/results
- **Visually distinct result cards**: Each result is shown as a card with scheme name, state, tags, relevance score, and expandable description
- **Helpful status messages**: See how many results were found, what filters are active, and get suggestions if no results are found
- Fast search using FAISS index (both L2 and Cosine)

## Setup Instructions

### 1. Install dependencies

It's recommended to use a virtual environment. Then install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Prepare Data and Index
- Ensure `myscheme_data.csv` and `faiss_index.bin` are present in the project directory.
- The cosine similarity index (`cosine_faiss_index.bin`) will be created automatically on first run if it does not exist.

### 3. Run the Streamlit Dashboard

```bash
streamlit run streamlit_dashboard.py
```

## Usage
- Enter your search query in the input box (e.g., "scholarship for students in Maharashtra").
- **Filter by state** using the sidebar dropdown for explicit control, or let the dashboard detect the state from your query.
- Click **Search** to see relevant schemes. Use **Clear Search** to reset your query and results.
- Results are shown in two columns:
  - **Left:** Euclidean (L2) distance results (lower score = more similar)
  - **Right:** Cosine similarity results (higher score = more similar)
- Each result card shows the scheme name, state, tags, a relevance score, and an expandable description.
- A summary below the results shows how many schemes appear in both searches.
- If no results are found, try different keywords or select a different state.

## How Similarity Search Works
- **L2 (Euclidean) Search:** Uses a FAISS index (`faiss_index.bin`) with L2 distance. Lower scores mean higher similarity.
- **Cosine Similarity Search:** Uses a FAISS index (`cosine_faiss_index.bin`) with normalized embeddings and inner product (dot product). Higher scores mean higher similarity.
- Both indices are managed automatically by the dashboard.

## Advanced: Cosine Similarity Utilities
- The module `cosine_faiss_utils.py` provides functions to create, save, load, and search a cosine similarity FAISS index.
- You can use these utilities in your own scripts for batch processing or experimentation.

## Requirements
- Python 3.8+
- See `requirements.txt` for Python package dependencies.

## Files
- `streamlit_dashboard.py`: Main Streamlit app (now with L2 vs Cosine comparison)
- `cosine_faiss_utils.py`: Utilities for cosine similarity FAISS index
- `myscheme_data.csv`: Scheme data
- `faiss_index.bin`: Precomputed FAISS index (L2)
- `cosine_faiss_index.bin`: Precomputed or auto-generated FAISS index (cosine similarity)
- `requirements.txt`: Python dependencies

---