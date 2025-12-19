"""Result ranking for hybrid retrieval."""
import yaml
from pathlib import Path
from typing import List, Tuple
import numpy as np


def load_config():
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class HybridRanker:
    """Rank and merge results from local and global search."""
    
    def __init__(self, alpha: float = None):
        """
        Initialize ranker with weighting parameter.
        
        Args:
            alpha: Weight for local search (1-alpha for global). Loaded from config if None.
        """
        if alpha is None:
            config = load_config()
            alpha = config.get('retrieval', {}).get('hybrid', {}).get('alpha', 0.6)
        self.alpha = alpha
    
    def rank(self, 
             local_scores: List[Tuple[int, float]], 
             global_scores: List[Tuple[int, float]]) -> List[int]:
        """
        Hybrid ranking combining local and global scores.
        
        Args:
            local_scores: List of (chunk_id, score) from local search
            global_scores: List of (chunk_id, score) from global search
            
        Returns:
            Ranked list of chunk IDs
        """
        # Normalize scores
        local_dict = self._normalize_scores(local_scores)
        global_dict = self._normalize_scores(global_scores)
        
        # Combine scores
        all_ids = set(local_dict.keys()) | set(global_dict.keys())
        combined = {}
        
        for idx in all_ids:
            local_score = local_dict.get(idx, 0.0)
            global_score = global_dict.get(idx, 0.0)
            combined[idx] = self.alpha * local_score + (1 - self.alpha) * global_score
        
        # Sort by combined score
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in ranked]
    
    def _normalize_scores(self, scores: List[Tuple[int, float]]) -> dict:
        """Normalize scores to [0, 1] range."""
        if not scores:
            return {}
        
        score_dict = {idx: score for idx, score in scores}
        values = list(score_dict.values())
        
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:
            return {idx: 1.0 for idx in score_dict.keys()}
        
        return {
            idx: (score - min_val) / (max_val - min_val)
            for idx, score in score_dict.items()
        }
