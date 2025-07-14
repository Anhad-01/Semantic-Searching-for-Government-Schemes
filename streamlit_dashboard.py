import streamlit as st
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import string
import re
from cosine_faiss_utils import load_cosine_index, create_and_save_cosine_index, search_cosine_index
from scheme_agents import euclidean_search_agent, cosine_search_agent, result_curator_agent

# Set Streamlit page config to wide layout
st.set_page_config(layout='wide')

def clean_text(text):
    if pd.isnull(text):
        return ''
    # Lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_resource
def load_assets():
    # Load the CSV data
    df = pd.read_csv('myscheme_data.csv')
    
    # Apply clean_text to relevant columns
    for col in ['name', 'description', 'tags']:
        cleaned_col = f'cleaned_{col}'
        df[cleaned_col] = df[col].apply(clean_text)

    # Create combined_text column
    combined_cols = ['cleaned_name', 'cleaned_description', 'cleaned_tags']
    df['combined_text'] = df[combined_cols].agg(' '.join, axis=1)
    
    # Load the SentenceTransformer model
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    # Load the L2 FAISS index
    l2_index = faiss.read_index('faiss_index.bin')
    
    # Load or create the cosine similarity index
    try:
        cosine_index = load_cosine_index('cosine_faiss_index.bin')
    except:
        # If cosine index doesn't exist, create it
        st.info("Creating cosine similarity index... This may take a moment.")
        # Generate embeddings for the combined text
        text_embeddings = model.encode(df['combined_text'].tolist(), show_progress_bar=True)
        cosine_index = create_and_save_cosine_index(text_embeddings, 'cosine_faiss_index.bin')
    
    return df, model, l2_index, cosine_index

df, model, l2_index, cosine_index = load_assets()

# Comprehensive list of Indian states and UTs
STATE_LIST = [
    'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana',
    'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',
    'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu',
    'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal', 'Andaman and Nicobar Islands',
    'Chandigarh', 'Dadra and Nagar Haveli and Daman and Diu', 'Delhi', 'Jammu and Kashmir', 'Ladakh',
    'Lakshadweep', 'Puducherry'
]

def extract_state_from_query(query, state_list):
    query_lower = query.lower()
    for state in state_list:
        if state.lower() in query_lower:
            return state  # Return original cased state name
    return None

# Function to get filtered schemes based on query and detected state (L2 and Cosine comparison)

def get_filtered_schemes_comparison(query, detected_state, df, model, l2_index, cosine_index, top_k=10, explicit_state=None):
    # Preprocess query
    cleaned_query = clean_text(query)
    # Embed query
    query_embedding = model.encode([cleaned_query])
    
    # L2 Search (Euclidean Distance)
    D_l2, I_l2 = l2_index.search(query_embedding, top_k * 10)
    top_indices_l2 = I_l2[0]
    top_scores_l2 = D_l2[0]
    results_df_l2 = df.iloc[top_indices_l2].copy()
    results_df_l2['relevance_score'] = top_scores_l2
    
    # Cosine Search (Cosine Similarity)
    cosine_scores, cosine_indices = search_cosine_index(query_embedding, cosine_index, top_k * 10)
    top_indices_cosine = cosine_indices[0]
    top_scores_cosine = cosine_scores[0]
    results_df_cosine = df.iloc[top_indices_cosine].copy()
    results_df_cosine['relevance_score'] = top_scores_cosine
    
    # Determine which state filter to use
    state_to_use = None
    if explicit_state and explicit_state != 'All States':
        state_to_use = explicit_state
    elif detected_state:
        state_to_use = detected_state
    
    # Apply filtering logic to both L2 and Cosine results
    def apply_state_filter(results_df, state_to_use):
        if state_to_use:
            filtered = results_df[
                results_df['state'].str.lower().eq(state_to_use.lower()) |
                results_df['state'].str.lower().str.contains('ministry')
            ]
            if filtered.empty:
                # If no results after filtering, return top_k unfiltered
                return filtered, results_df[['name', 'state', 'description', 'tags', 'relevance_score']].head(top_k), True
            else:
                return filtered[['name', 'state', 'description', 'tags', 'relevance_score']].head(top_k), None, False
        else:
            # No state detected or selected, return top_k unfiltered
            return None, results_df[['name', 'state', 'description', 'tags', 'relevance_score']].head(top_k), None
    
    # Apply filtering to L2 results
    l2_filtered, l2_fallback, l2_fallback_used = apply_state_filter(results_df_l2, state_to_use)
    if l2_filtered is not None and not l2_filtered.empty:
        l2_results = l2_filtered
    else:
        l2_results = l2_fallback
    
    # Apply filtering to Cosine results
    cosine_filtered, cosine_fallback, cosine_fallback_used = apply_state_filter(results_df_cosine, state_to_use)
    if cosine_filtered is not None and not cosine_filtered.empty:
        cosine_results = cosine_filtered
    else:
        cosine_results = cosine_fallback
    
    return l2_results, cosine_results, state_to_use, l2_fallback_used, cosine_fallback_used

# Initialize session state for query and results
if 'query' not in st.session_state:
    st.session_state.query = ''
if 'l2_results' not in st.session_state:
    st.session_state.l2_results = None
if 'cosine_results' not in st.session_state:
    st.session_state.cosine_results = None
if 'detected_state' not in st.session_state:
    st.session_state.detected_state = None
if 'search_performed' not in st.session_state:
    st.session_state.search_performed = False

# Get unique states for the filter (sorted, with 'All States' at the top)
unique_states = sorted(df['state'].dropna().unique())
state_options = ['All States'] + unique_states

# Main title and description
st.title('Semantic Search for Government Schemes')
# Add introductory section
st.markdown(
    'This dashboard allows you to search for government schemes and compare results using Euclidean and Cosine Similarity. You can also filter by state or Ministry.'
)
st.divider()

# Search Parameters section
st.subheader('Search Parameters')
with st.container():
    st.markdown('**Filter by State or Ministry:**')
    state_col = st.columns([1])[0]
    selected_state = state_col.selectbox(
        '',
        state_options,
        index=0,
        key='state_filter',
    )
    # Search form for query input and search button
    with st.form("search_form"):
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            user_query = st.text_input(
                'Enter your search query:',
                value='',
                placeholder='e.g., education schemes in Maharashtra, farmer loans',
                key='main_query',
            )
        with col2:
            submitted = st.form_submit_button('Search', type='primary')

st.divider()

# Handle Search and Results Display
if submitted and user_query.strip():
    st.write('Searching...')
    detected_state = extract_state_from_query(user_query, STATE_LIST)
    # Use CrewAI agents for search
    l2_results = euclidean_search_agent.tools[0].run(query=user_query, df=df, model=model, l2_index=l2_index, detected_state=detected_state, top_k=10)
    cosine_results = cosine_search_agent.tools[0].run(query=user_query, df=df, model=model, cosine_index=cosine_index, detected_state=detected_state, top_k=10)
    # Curate results
    l2_results = result_curator_agent.tools[0].run(search_results_df=l2_results, detected_state=detected_state)
    cosine_results = result_curator_agent.tools[0].run(search_results_df=cosine_results, detected_state=detected_state)
    state_used = detected_state if selected_state == 'All States' else selected_state
    # Display state filter information
    if state_used:
        if selected_state != 'All States' and selected_state == state_used:
            st.info(f"Explicit filter: Showing results for state: {state_used.title()} (including central schemes)")
        else:
            st.info(f"Detected from query: Showing results for state: {state_used.title()} (including central schemes)")
    # Create two columns for side-by-side comparison
    col1, col2 = st.columns(2)
    # Left column: L2 (Euclidean Distance) results
    with col1:
        st.subheader("Results (Euclidean Distance)")
        if l2_results is not None and not l2_results.empty:
            st.success(f"Found {len(l2_results)} result(s).")
            # If a state filter is applied, sort state schemes first, then ministry schemes, both by score
            if state_used:
                state_mask = l2_results['state'].str.lower() == state_used.lower()
                ministry_mask = l2_results['state'].str.lower().str.contains('ministry')
                state_schemes = l2_results[state_mask].sort_values('relevance_score', ascending=False)
                ministry_schemes = l2_results[ministry_mask].sort_values('relevance_score', ascending=False)
                other_schemes = l2_results[~(state_mask | ministry_mask)].sort_values('relevance_score', ascending=False)
                display_df = pd.concat([state_schemes, ministry_schemes, other_schemes])
            else:
                display_df = l2_results.sort_values('relevance_score', ascending=False)
            for idx, row in display_df.iterrows():
                st.markdown(f"#### {row['name']}")
                st.write(f"**State:** {row['state']}")
                try:
                    tags_list = eval(row['tags']) if isinstance(row['tags'], str) else row['tags']
                    if isinstance(tags_list, list):
                        tags = ', '.join([t.strip() for t in tags_list if t.strip()])
                    else:
                        tags = str(row['tags'])
                except:
                    tags = str(row['tags'])
                if tags:
                    st.write(f"**Tags:** {tags}")
                if 'relevance_score' in row:
                    st.write(f"**L2 Distance Score:** {row['relevance_score']:.4f}")
                with st.expander("Description", expanded=False):
                    st.write(row['description'])
                st.divider()
        else:
            st.info('No schemes found for L2 (Euclidean Distance) search.')
    # Right column: Cosine Similarity results
    with col2:
        st.subheader("Results (Cosine Similarity)")
        if cosine_results is not None and not cosine_results.empty:
            st.success(f"Found {len(cosine_results)} result(s).")
            # If a state filter is applied, sort state schemes first, then ministry schemes, both by score
            if state_used:
                state_mask = cosine_results['state'].str.lower() == state_used.lower()
                ministry_mask = cosine_results['state'].str.lower().str.contains('ministry')
                state_schemes = cosine_results[state_mask].sort_values('relevance_score', ascending=False)
                ministry_schemes = cosine_results[ministry_mask].sort_values('relevance_score', ascending=False)
                other_schemes = cosine_results[~(state_mask | ministry_mask)].sort_values('relevance_score', ascending=False)
                display_df = pd.concat([state_schemes, ministry_schemes, other_schemes])
            else:
                display_df = cosine_results.sort_values('relevance_score', ascending=False)
            for idx, row in display_df.iterrows():
                st.markdown(f"#### {row['name']}")
                st.write(f"**State:** {row['state']}")
                try:
                    tags_list = eval(row['tags']) if isinstance(row['tags'], str) else row['tags']
                    if isinstance(tags_list, list):
                        tags = ', '.join([t.strip() for t in tags_list if t.strip()])
                    else:
                        tags = str(row['tags'])
                except:
                    tags = str(row['tags'])
                if tags:
                    st.write(f"**Tags:** {tags}")
                if 'relevance_score' in row:
                    st.write(f"**Cosine Similarity Score:** {row['relevance_score']:.4f}")
                with st.expander("Description", expanded=False):
                    st.write(row['description'])
                st.divider()
        else:
            st.info('No schemes found for Cosine Similarity search.')
    # Add comparison summary
    if (l2_results is not None and not l2_results.empty) and (cosine_results is not None and not cosine_results.empty):
        st.subheader("Comparison Summary")
        l2_names = set(l2_results['name'].tolist())
        cosine_names = set(cosine_results['name'].tolist())
        common_results = len(l2_names & cosine_names)
        st.write(f"**Common results between L2 and Cosine:** {common_results}/{len(l2_results)}")
        if common_results > 0:
            st.write("**Schemes appearing in both searches:**")
            for name in sorted(l2_names & cosine_names):
                st.write(f"- {name}")