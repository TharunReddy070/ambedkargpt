"""
Semantic Chunker implementing Algorithm 1 from SEMRAG paper.

Uses cosine similarity of sentence embeddings to group sentences into 
semantically coherent chunks with token limit enforcement.
"""
import yaml
import nltk
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from .buffer_merger import BufferMerger

nltk.download("punkt", quiet=True)


def load_config():
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class SemanticChunker:
    """
    Semantic chunking via cosine similarity (Algorithm 1).
    
    Enforces token limits:
    - Max chunk size: 1024 tokens
    - Creates sub-chunks of ~128 tokens with overlap when needed
    """
    
    def __init__(
        self,
        model_name: str = None,
        threshold: float = None,
        buffer_size: int = None,
        max_tokens: int = None,
        subchunk_tokens: int = None,
        overlap_tokens: int = None
    ):
        # Load from config if not provided
        config = load_config()
        chunking_config = config.get('chunking', {})
        embeddings_config = config.get('embeddings', {})
        
        model_name = model_name or embeddings_config.get('model', 'all-MiniLM-L6-v2')
        threshold = threshold if threshold is not None else chunking_config.get('similarity_threshold', 0.3)
        buffer_size = buffer_size if buffer_size is not None else chunking_config.get('buffer_size', 2)
        max_tokens = max_tokens if max_tokens is not None else chunking_config.get('max_tokens', 1024)
        subchunk_tokens = subchunk_tokens if subchunk_tokens is not None else chunking_config.get('subchunk_tokens', 128)
        overlap_tokens = overlap_tokens if overlap_tokens is not None else chunking_config.get('overlap_tokens', 20)
        
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self.buffer_merger = BufferMerger(buffer_size)
        self.max_tokens = max_tokens
        self.subchunk_tokens = subchunk_tokens
        self.overlap_tokens = overlap_tokens

    def _count_tokens(self, text: str) -> int:
        """Approximate token count (whitespace split)."""
        return len(text.split())
    
    def _split_into_subchunks(self, text: str) -> list[str]:
        """
        Split large chunk into sub-chunks with overlap.
        
        Creates ~128 token sub-chunks with 20 token overlap for context continuity.
        """
        words = text.split()
        subchunks = []
        
        start = 0
        while start < len(words):
            end = min(start + self.subchunk_tokens, len(words))
            subchunk = " ".join(words[start:end])
            subchunks.append(subchunk)
            
            # Move start with overlap
            start = end - self.overlap_tokens if end < len(words) else end
        
        return subchunks

    def chunk(self, text: str) -> list[str]:
        """
        Semantic chunking with token limit enforcement.
        
        Algorithm:
        1. Split text into sentences
        2. Apply buffer merging for context
        3. Group by cosine similarity (Algorithm 1)
        4. Enforce max 1024 token limit
        5. Create ~128 token sub-chunks with overlap if needed
        
        Args:
            text: Input text
            
        Returns:
            List of semantic chunks respecting token limits
        """
        sentences = nltk.sent_tokenize(text)
        if not sentences:
            return []
        
        buffered = self.buffer_merger.merge(sentences)
        embeddings = self.model.encode(buffered)

        chunks = []
        current = [sentences[0]]

        for i in range(len(embeddings) - 1):
            similarity = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i + 1].reshape(1, -1),
            )[0][0]

            if 1 - similarity < self.threshold:
                # Check token limit before adding
                potential = " ".join(current + [sentences[i + 1]])
                if self._count_tokens(potential) <= self.max_tokens:
                    current.append(sentences[i + 1])
                else:
                    # Token limit reached, finalize chunk
                    chunks.append(" ".join(current))
                    current = [sentences[i + 1]]
            else:
                # Low similarity, create new chunk
                chunks.append(" ".join(current))
                current = [sentences[i + 1]]

        # Add final chunk
        if current:
            chunks.append(" ".join(current))
        
        # Split chunks exceeding max_tokens into sub-chunks with overlap
        final_chunks = []
        for chunk in chunks:
            if self._count_tokens(chunk) > self.max_tokens:
                final_chunks.extend(self._split_into_subchunks(chunk))
            else:
                final_chunks.append(chunk)
        
        return final_chunks
