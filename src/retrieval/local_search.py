"""
Local Graph RAG Search (Equation 4 from SEMRAG paper).

Entity-focused retrieval for specific, narrow questions.
"""
import yaml
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


def load_config():
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class LocalGraphSearch:
    """
    Implement Equation 4: Local entity-focused retrieval.
    
    D_retrieved = Top_k({v ∈ V, g ∈ G | sim(v, Q+H) > τ_e ∧ sim(g, v) > τ_d})
    
    Steps:
    1. Calculate similarity between query and entities
    2. Filter by threshold τ_e
    3. Find chunks related to those entities
    4. Filter by threshold τ_d
    5. Return top_k results with scores
    """
    
    def __init__(self, embedder):
        self.embedder = embedder

    def search(
        self,
        query: str,
        entity_embeddings: dict,
        chunk_embeddings: dict,
        tau_e: float = 0.3,
        tau_d: float = 0.3,
        top_k: int = 10,
    ) -> list[int]:
        """Return chunk IDs only (backward compatibility)."""
        results_with_scores = self.search_with_scores(
            query, entity_embeddings, chunk_embeddings, tau_e, tau_d, top_k
        )
        return [cid for cid, _ in results_with_scores]
    
    def search_with_scores(
        self,
        query: str,
        entity_embeddings: dict,
        chunk_embeddings: dict,
        tau_e: float = None,
        tau_d: float = None,
        top_k: int = None,
    ) -> list[tuple[int, float]]:
        """
        Search with similarity scores for re-ranking.
        
        Returns:
            List of (chunk_id, score) tuples
        """
        # Load from config if not provided
        config = load_config()
        local_config = config.get('retrieval', {}).get('local', {})
        
        tau_e = tau_e if tau_e is not None else local_config.get('tau_e', 0.3)
        tau_d = tau_d if tau_d is not None else local_config.get('tau_d', 0.3)
        top_k = top_k if top_k is not None else local_config.get('top_k', 10)
        
        query_emb = self.embedder.encode(query)
        chunk_scores = {}

        # Step 1-2: Find relevant entities above threshold τ_e
        for entity, entity_emb in entity_embeddings.items():
            entity_sim = cosine_similarity([query_emb], [entity_emb])[0][0]
            
            if entity_sim > tau_e:
                # Step 3-4: Find chunks related to entity above threshold τ_d
                for cid, chunk_emb in chunk_embeddings.items():
                    chunk_entity_sim = cosine_similarity([entity_emb], [chunk_emb])[0][0]
                    
                    if chunk_entity_sim > tau_d:
                        # Combine scores: entity relevance + chunk-entity relation
                        combined_score = entity_sim * chunk_entity_sim
                        
                        # Keep highest score for each chunk
                        if cid not in chunk_scores or combined_score > chunk_scores[cid]:
                            chunk_scores[cid] = combined_score
        
        # Step 5: Sort and return top-K with scores
        results = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        return results[:top_k]
