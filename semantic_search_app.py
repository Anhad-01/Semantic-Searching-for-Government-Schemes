import subprocess
import sys
import re

# List of required packages
required_packages = [
    'pandas',
    'scikit-learn',
    'sentence-transformers',
    'faiss-cpu'
]

def install_if_missing(package):
    try:
        __import__(package if package != 'faiss-cpu' else 'faiss')
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

for pkg in required_packages:
    install_if_missing(pkg)

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

# Load the CSV file into a DataFrame
csv_path = 'myscheme_data.csv'
df = pd.read_csv(csv_path)

# Display the first 5 rows
print("First 5 rows of the DataFrame:")
print(df.head())

# Display column information
print("\nDataFrame column info:")
df.info()

# Text preprocessing function
def clean_text(text):
    """
    Lowercase, remove punctuation/special characters, and extra whitespace.
    For production: add language-specific normalization (e.g., Hinglish transliteration/correction) here.
    """
    if not isinstance(text, str):
        return ''
    # Lowercase
    text = text.lower()
    # Remove punctuation and special characters (keep alphanumeric and spaces)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply clean_text to relevant columns
for col in ['name', 'description', 'tags']:
    cleaned_col = f'cleaned_{col}'
    df[cleaned_col] = df[col].apply(clean_text)

# Display the first 5 rows with new cleaned columns
print("\nFirst 5 rows with cleaned columns:")
print(df[[
    'name', 'cleaned_name',
    'description', 'cleaned_description',
    'tags', 'cleaned_tags'
]].head())

# Load the multilingual Sentence Transformer model
model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
print(f"Loading SentenceTransformer model: {model_name}")
model = SentenceTransformer(model_name)

# Combine cleaned columns into a single string
combine_cols = ['cleaned_name', 'cleaned_description', 'cleaned_tags']
df['combined_text'] = df[combine_cols].fillna('').agg(' '.join, axis=1)

# Generate embeddings for the combined_text column
print("Generating embeddings for combined_text...")
text_embeddings = model.encode(df['combined_text'].tolist(), show_progress_bar=True)

# Print the shape of the embeddings array
print(f"\nShape of text_embeddings: {text_embeddings.shape}")

# Get the dimensionality of the embeddings
embedding_dim = text_embeddings.shape[1]

# Initialize a FAISS index (L2 distance)
index = faiss.IndexFlatL2(embedding_dim)

# Ensure embeddings are float32
if text_embeddings.dtype != 'float32':
    text_embeddings = text_embeddings.astype('float32')

# Add embeddings to the index
index.add(text_embeddings)

# Print the number of vectors in the index
print(f"\nNumber of vectors in FAISS index: {index.ntotal}")

# (Optional) Save the FAISS index and DataFrame mapping for persistence
faiss.write_index(index, 'faiss_index.bin')
print("FAISS index saved to faiss_index.bin")

# Save DataFrame mapping (id and name columns for retrieval)
df[['name', 'description', 'tags', 'combined_text']].to_csv('scheme_mapping.csv', index=False)
print("Scheme mapping saved to scheme_mapping.csv")

def search_schemes(query, model, index, df, top_k=10):
    """
    Search for the most similar schemes to the query using FAISS and SentenceTransformer.
    """
    # Preprocess the query
    cleaned_query = clean_text(query)
    # Generate embedding for the query
    query_embedding = model.encode([cleaned_query])
    if query_embedding.dtype != 'float32':
        query_embedding = query_embedding.astype('float32')
    # Search FAISS index
    D, I = index.search(query_embedding, top_k)
    # Retrieve and print results
    print(f"\nTop {top_k} results for query: '{query}'\n")
    for rank, idx in enumerate(I[0]):
        if idx < 0 or idx >= len(df):
            continue
        row = df.iloc[idx]
        print(f"Result {rank+1}:")
        print(f"  Name: {row.get('name', 'N/A')}")
        print(f"  State: {row.get('state', 'N/A')}")
        print(f"  Description: {row.get('description', 'N/A')}")
        print("-")

if __name__ == "__main__":
    while True:
        user_query = input("Enter your query (or 'quit' to exit): ").strip()
        if user_query.lower() == 'quit':
            print("Exiting search.")
            break
        search_schemes(user_query, model, index, df, top_k=10) 