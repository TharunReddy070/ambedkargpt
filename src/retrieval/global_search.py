"""
Global Graph RAG Search (Equation 5 from SEMRAG paper).

Community-focused retrieval for broad, conceptual questions.
"""
import yaml
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


def load_config():
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class GlobalGraphSearch:
    """
    Implement Equation 5: Global community-focused retrieval.
    
    D_retrieved = Top_k(⋃_{r ∈ R_Top-K(Q)} ⋃_{c_i ∈ C_r} (⋃_{p_j ∈ c_i} 
                       (p_j, score(p_j, Q))), score(p_j, Q))
    
    Steps:
    1. Find top-K community reports relevant to query
    2. Extract chunks from those communities
    3. Score each point within chunks
    4. Return top-K points based on scores
    """
    
    def __init__(self, embedder):
        self.embedder = embedder

    def search(
        self,
        query: str,
        community_summaries: dict,
        top_k: int = 3,
    ) -> list[int]:
        """Return community IDs only (backward compatibility)."""
        results_with_scores = self.search_with_scores(query, community_summaries, top_k)
        return [cid for cid, _ in results_with_scores]
    
    def search_with_scores(
        self,
        query: str,
        community_summaries: dict,
        top_k: int = None,
    ) -> list[tuple[int, float]]:
        """
        Search with similarity scores for re-ranking.
        
        Returns:
            List of (community_id, score) tuples
        """
        # Load from config if not provided
        config = load_config()
        global_config = config.get('retrieval', {}).get('global', {})
        
        top_k = top_k if top_k is not None else global_config.get('top_k', 5)
        
        query_emb = self.embedder.encode(query)
        scores = []

        # Step 1: Calculate similarity between query and community summaries
        for cid, summary in community_summaries.items():
            summary_emb = self.embedder.encode(summary)
            score = cosine_similarity([query_emb], [summary_emb])[0][0]
            scores.append((cid, score))

        # Step 4: Sort and return top-K communities with scores
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
