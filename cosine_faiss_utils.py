import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def create_and_save_cosine_index(embeddings, index_path='cosine_faiss_index.bin'):
    """
    Create and save a FAISS index optimized for cosine similarity.
    
    This function normalizes embeddings to unit length and creates an IndexFlatIP
    which computes dot products (equivalent to cosine similarity for normalized vectors).
    
    Parameters:
    -----------
    embeddings : numpy.ndarray
        Pre-computed embeddings of shape (n_vectors, embedding_dim)
    index_path : str, optional
        Path where the FAISS index will be saved (default: 'cosine_faiss_index.bin')
    
    Returns:
    --------
    faiss.IndexFlatIP
        The created FAISS index
    """
    # Ensure embeddings are float32
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    
    # Normalize embeddings to unit length (L2 normalization)
    # This is crucial for cosine similarity with IndexFlatIP
    embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Get the dimensionality of the embeddings
    embedding_dim = embeddings_normalized.shape[1]
    
    # Initialize a FAISS index for Inner Product (dot product)
    # This computes dot products, which are equivalent to cosine similarity for normalized vectors
    index = faiss.IndexFlatIP(embedding_dim)
    
    # Add the normalized embeddings to the index
    index.add(embeddings_normalized)
    
    # Save the index to the specified path
    faiss.write_index(index, index_path)
    
    # Print confirmation message
    print(f"Cosine similarity FAISS index created and saved to: {index_path}")
    print(f"Number of vectors in index: {index.ntotal}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Index type: {type(index).__name__}")
    
    return index


def load_cosine_index(index_path='cosine_faiss_index.bin'):
    """
    Load a FAISS index from the specified path.
    
    Parameters:
    -----------
    index_path : str, optional
        Path to the FAISS index file (default: 'cosine_faiss_index.bin')
    
    Returns:
    --------
    faiss.Index
        The loaded FAISS index
    """
    try:
        index = faiss.read_index(index_path)
        print(f"Cosine similarity FAISS index loaded from: {index_path}")
        print(f"Number of vectors in index: {index.ntotal}")
        print(f"Index type: {type(index).__name__}")
        return index
    except Exception as e:
        print(f"Error loading index from {index_path}: {e}")
        raise


def search_cosine_index(query_embedding, index, top_k):
    """
    Search the cosine similarity index for the most similar vectors.
    
    This function normalizes the query embedding and performs a search
    using the IndexFlatIP, returning cosine similarity scores.
    
    Parameters:
    -----------
    query_embedding : numpy.ndarray
        Query embedding of shape (1, embedding_dim) or (embedding_dim,)
    index : faiss.Index
        The FAISS index to search in
    top_k : int
        Number of top results to return
    
    Returns:
    --------
    tuple
        (distances, indices) where:
        - distances: numpy.ndarray of shape (1, top_k) containing cosine similarity scores
        - indices: numpy.ndarray of shape (1, top_k) containing the indices of similar vectors
    
    Note:
    -----
    - Higher scores indicate higher similarity (cosine similarity ranges from -1 to 1)
    - For normalized vectors, the dot product equals cosine similarity
    """
    # Ensure query_embedding is a 2D array
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    # Ensure query_embedding is float32
    if query_embedding.dtype != np.float32:
        query_embedding = query_embedding.astype(np.float32)
    
    # Normalize the query embedding to unit length
    query_normalized = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    
    # Perform the search
    # For IndexFlatIP, higher scores indicate higher similarity
    distances, indices = index.search(query_normalized, top_k)
    
    return distances, indices


def create_cosine_index_from_texts(texts, model_name='paraphrase-multilingual-MiniLM-L12-v2', 
                                  index_path='cosine_faiss_index.bin'):
    """
    Convenience function to create embeddings from texts and build a cosine similarity index.
    
    Parameters:
    -----------
    texts : list
        List of text strings to embed
    model_name : str, optional
        Name of the SentenceTransformer model to use (default: 'paraphrase-multilingual-MiniLM-L12-v2')
    index_path : str, optional
        Path where the FAISS index will be saved (default: 'cosine_faiss_index.bin')
    
    Returns:
    --------
    faiss.IndexFlatIP
        The created FAISS index
    """
    print(f"Loading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print("Generating embeddings from texts...")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Create and save the cosine similarity index
    index = create_and_save_cosine_index(embeddings, index_path)
    
    return index


def compare_l2_vs_cosine_scores(query_embedding, l2_index, cosine_index, top_k=10):
    """
    Compare results between L2 distance and cosine similarity searches.
    
    Parameters:
    -----------
    query_embedding : numpy.ndarray
        Query embedding
    l2_index : faiss.Index
        L2 distance FAISS index
    cosine_index : faiss.Index
        Cosine similarity FAISS index
    top_k : int, optional
        Number of top results to compare (default: 10)
    
    Returns:
    --------
    dict
        Dictionary containing comparison results
    """
    # L2 search (lower scores = more similar)
    l2_distances, l2_indices = l2_index.search(query_embedding.reshape(1, -1).astype(np.float32), top_k)
    
    # Cosine search (higher scores = more similar)
    cosine_distances, cosine_indices = search_cosine_index(query_embedding, cosine_index, top_k)
    
    comparison = {
        'l2_distances': l2_distances[0],
        'l2_indices': l2_indices[0],
        'cosine_distances': cosine_distances[0],
        'cosine_indices': cosine_indices[0],
        'common_indices': set(l2_indices[0]) & set(cosine_indices[0])
    }
    
    print(f"L2 vs Cosine Similarity Comparison (top {top_k} results):")
    print(f"Common results: {len(comparison['common_indices'])}/{top_k}")
    print(f"L2 distance range: {l2_distances[0].min():.4f} - {l2_distances[0].max():.4f}")
    print(f"Cosine similarity range: {cosine_distances[0].min():.4f} - {cosine_distances[0].max():.4f}")
    
    return comparison 