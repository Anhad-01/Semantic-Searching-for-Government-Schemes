from crewai import Agent, Crew
from crewai.tools import BaseTool
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from cosine_faiss_utils import search_cosine_index
from pydantic import BaseModel, Field
from typing import Optional


# --- Parameter Schemas ---
class SearchParams(BaseModel):
    query: str = Field(description="The search query text")
    df: object = Field(description="The pandas DataFrame containing scheme data")
    model: object = Field(description="The SentenceTransformer model")
    l2_index: object = Field(description="The FAISS index for search")
    detected_state: Optional[str] = Field(None, description="Detected state from query, if any")
    top_k: int = Field(default=10, description="Number of results to return")

class CosineParams(BaseModel):
    query: str = Field(description="The search query text")
    df: object = Field(description="The pandas DataFrame containing scheme data")
    model: object = Field(description="The SentenceTransformer model")
    cosine_index: object = Field(description="The cosine similarity FAISS index")
    detected_state: Optional[str] = Field(None, description="Detected state from query, if any")
    top_k: int = Field(default=10, description="Number of results to return")

class CuratorParams(BaseModel):
    search_results_df: object = Field(description="DataFrame containing search results to curate")
    detected_state: Optional[str] = Field(None, description="Detected state from query, if any")


# --- Tool Definitions ---
class EuclideanSearchTool(BaseTool):
    name: str = "Euclidean Searcher"
    description: str = "Performs semantic search using Euclidean distance and returns initial results."
    args_schema: type[BaseModel] = SearchParams
    
    def _run(self, query: str, df: object, model: object, l2_index: object, detected_state: Optional[str] = None, top_k: int = 10):
        cleaned_query = query.lower().strip()
        query_embedding = model.encode([cleaned_query])
        D_l2, I_l2 = l2_index.search(query_embedding, top_k)
        top_indices_l2 = I_l2[0]
        top_scores_l2 = D_l2[0]
        results_df_l2 = df.iloc[top_indices_l2].copy()
        results_df_l2['relevance_score'] = top_scores_l2
        return results_df_l2

class CosineSearchTool(BaseTool):
    name: str = "Cosine Searcher"
    description: str = "Performs semantic search using Cosine similarity and returns initial results."
    args_schema: type[BaseModel] = CosineParams
    
    def _run(self, query: str, df: object, model: object, cosine_index: object, detected_state: Optional[str] = None, top_k: int = 10):
        cleaned_query = query.lower().strip()
        query_embedding = model.encode([cleaned_query])
        cosine_scores, cosine_indices = search_cosine_index(query_embedding, cosine_index, top_k)
        top_indices_cosine = cosine_indices[0]
        top_scores_cosine = cosine_scores[0]
        results_df_cosine = df.iloc[top_indices_cosine].copy()
        results_df_cosine['relevance_score'] = top_scores_cosine
        return results_df_cosine

class ResultCuratorTool(BaseTool):
    name: str = "Result Curator"
    description: str = "Sorts and prioritizes search results based on detected state and ministry status."
    args_schema: type[BaseModel] = CuratorParams
    
    def _run(self, search_results_df: object, detected_state: Optional[str] = None):
        if detected_state:
            state_mask = search_results_df['state'].str.lower() == detected_state.lower()
            ministry_mask = search_results_df['state'].str.lower().str.contains('ministry')
            state_schemes = search_results_df[state_mask].sort_values('relevance_score', ascending=False)
            ministry_schemes = search_results_df[ministry_mask].sort_values('relevance_score', ascending=False)
            other_schemes = search_results_df[~(state_mask | ministry_mask)].sort_values('relevance_score', ascending=False)
            display_df = pd.concat([state_schemes, ministry_schemes, other_schemes])
        else:
            display_df = search_results_df.sort_values('relevance_score', ascending=False)
        return display_df

euclidean_search_tool = EuclideanSearchTool()
cosine_search_tool = CosineSearchTool()
result_curator_tool = ResultCuratorTool()

# --- Agent Definitions ---
euclidean_search_agent = Agent(
    role='Euclidean Distance Scheme Search Expert',
    goal='Accurately find schemes using Euclidean distance based on query.',
    backstory='An expert in rapid information retrieval using L2 distance metrics.',
    tools=[euclidean_search_tool],
    verbose=True
)

cosine_search_agent = Agent(
    role='Cosine Similarity Scheme Search Specialist',
    goal='Identify schemes with high semantic relevance using Cosine similarity.',
    backstory='Highly skilled in semantic matching and understanding contextual relationships.',
    tools=[cosine_search_tool],
    verbose=True
)

result_curator_agent = Agent(
    role='Scheme Result Prioritizer and Sorter',
    goal='Organize and rank search results, prioritizing state-specific and ministry schemes.',
    backstory='A meticulous organizer, ensuring the most relevant and targeted schemes are presented first.',
    tools=[result_curator_tool],
    verbose=True
)

# --- (Optional) Crew Definition ---
# crew = Crew(
#     agents=[euclidean_search_agent, cosine_search_agent, result_curator_agent],
#     tasks=[],  # Define tasks as needed
#     verbose=True
# )

# --- (Optional) Task Definitions ---
# Example:
# from crewai import Task
# l2_task = Task(description='Perform initial Euclidean search...', agent=euclidean_search_agent)
# cosine_task = Task(description='Perform initial Cosine search...', agent=cosine_search_agent)
# curate_task = Task(description='Curate and sort results...', agent=result_curator_agent)

# The agents and tools are now ready to be imported and used in your main app.
