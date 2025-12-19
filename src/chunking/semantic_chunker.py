import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from .buffer_merger import BufferMerger

nltk.download("punkt")


class SemanticChunker:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.3,
        buffer_size: int = 2,
    ):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self.buffer_merger = BufferMerger(buffer_size)

    def chunk(self, text: str) -> list[str]:
        sentences = nltk.sent_tokenize(text)
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
                current.append(sentences[i + 1])
            else:
                chunks.append(" ".join(current))
                current = [sentences[i + 1]]

        chunks.append(" ".join(current))
        return chunks
