"""Unit tests for semantic chunking module."""
import unittest
from src.chunking.semantic_chunker import SemanticChunker


class TestSemanticChunking(unittest.TestCase):
    """Test semantic chunking implementation (Algorithm 1)."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.chunker = SemanticChunker()
        self.sample_text = """
        Dr. B.R. Ambedkar was a social reformer. He fought against caste discrimination.
        He was the principal architect of the Indian Constitution. The Constitution guarantees equality.
        Ambedkar believed in social justice. He advocated for the rights of marginalized communities.
        """
    
    def test_chunker_initialization(self):
        """Test chunker initializes correctly."""
        self.assertIsNotNone(self.chunker)
        self.assertTrue(hasattr(self.chunker, 'chunk'))
    
    def test_chunks_created(self):
        """Test that chunks are created from text."""
        chunks = self.chunker.chunk(self.sample_text)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
    
    def test_chunks_not_empty(self):
        """Test that chunks contain text."""
        chunks = self.chunker.chunk(self.sample_text)
        for chunk in chunks:
            self.assertIsInstance(chunk, str)
            self.assertGreater(len(chunk), 0)
    
    def test_semantic_similarity(self):
        """Test semantic grouping of related sentences."""
        chunks = self.chunker.chunk(self.sample_text)
        # Chunks should group semantically related content
        self.assertLess(len(chunks), 6)  # Should merge some sentences


if __name__ == '__main__':
    unittest.main()
