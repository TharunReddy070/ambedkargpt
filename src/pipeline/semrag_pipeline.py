from sentence_transformers import SentenceTransformer
from src.chunking.semantic_chunker import SemanticChunker
from src.graph.entity_extractor import EntityExtractor
from src.graph.graph_builder import KnowledgeGraphBuilder
from src.graph.community_detector import CommunityDetector
from src.graph.summarizer import CommunitySummarizer


class SemRAGPipeline:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.chunker = SemanticChunker()
        self.extractor = EntityExtractor()
        self.graph_builder = KnowledgeGraphBuilder()

    def build(self, text: str):
        chunks = self.chunker.chunk(text)
        chunk_embeddings = {}
        entity_embeddings = {}

        for idx, chunk in enumerate(chunks):
            entities = self.extractor.extract(chunk)
            self.graph_builder.add_chunk(idx, entities)

            chunk_embeddings[idx] = self.embedder.encode(chunk)
            for entity in entities:
                if entity not in entity_embeddings:
                    entity_embeddings[entity] = self.embedder.encode(entity)

        graph = self.graph_builder.get_graph()
        communities = CommunityDetector().detect(graph)

        summaries = {
            cid: CommunitySummarizer().summarize(entities)
            for cid, entities in communities.items()
        }

        return chunks, graph, chunk_embeddings, entity_embeddings, summaries
