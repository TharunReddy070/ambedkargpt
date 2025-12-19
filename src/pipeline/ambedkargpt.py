"""
Main SEMRAG pipeline for AmbedkarGPT.

Implements the complete SEMRAG system as per the research paper:
1. Semantic Chunking (Algorithm 1) - Cosine similarity-based chunking
2. Knowledge Graph Construction - Entity extraction and relationship building
3. Community Detection - Leiden/Louvain algorithm for entity grouping
4. Dual Retrieval - Local (Equation 4) and Global (Equation 5) search

Outputs are saved to data/processed/:
- chunks.json: Semantic chunks with metadata
- knowledge_graph.pkl: NetworkX graph with entities, relationships, communities
"""
import json
import pickle
import os
from pathlib import Path
import pypdf
from sentence_transformers import SentenceTransformer
from src.chunking.semantic_chunker import SemanticChunker
from src.graph.entity_extractor import EntityExtractor
from src.graph.graph_builder import KnowledgeGraphBuilder
from src.graph.community_detector import CommunityDetector
from src.graph.summarizer import CommunitySummarizer


def load_pdf_text(pdf_path: str) -> str:
    """
    Extract text from PDF file using pypdf.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted text as single string
    """
    reader = pypdf.PdfReader(pdf_path)
    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text())
    return "\n".join(text_parts)


class SemRAGPipeline:
    """
    Main pipeline implementing SEMRAG paper methodology.
    
    Steps:
    1. Semantic chunking via cosine similarity (Algorithm 1)
    2. Entity extraction using spaCy NER
    3. Knowledge graph construction (nodes=entities, edges=relationships)
    4. Community detection using Louvain algorithm
    5. Community summarization using LLM
    6. Save outputs to data/processed/
    """
    
    def __init__(self, output_dir: str = "data/processed"):
        """
        Initialize pipeline with embedding model and processors.
        
        Args:
            output_dir: Directory to save chunks.json and knowledge_graph.pkl
        """
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.chunker = SemanticChunker()
        self.extractor = EntityExtractor()
        self.graph_builder = KnowledgeGraphBuilder()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build(self, text: str):
        """
        Build complete knowledge graph from text.
        
        Implements SEMRAG paper Sections 3.2.2-3.2.3:
        1. Semantic chunking (Algorithm 1)
        2. Entity extraction from each chunk
        3. Relationship extraction between entities
        4. Graph construction (nodes=entities, edges=relationships)
        5. Community detection (Louvain algorithm)
        6. Community summarization via LLM
        
        Args:
            text: Input text from PDF
            
        Returns:
            Tuple of (chunks, graph, chunk_embeddings, entity_embeddings, summaries)
        """
        # Step 1: Semantic Chunking (Algorithm 1 from SEMRAG paper)
        chunks = self.chunker.chunk(text)
        
        # Step 2: Generate embeddings for retrieval
        chunk_embeddings = {}
        entity_embeddings = {}

        # Step 3: Entity extraction and graph construction
        for idx, chunk in enumerate(chunks):
            # Extract entities from chunk using spaCy NER
            entities = self.extractor.extract(chunk)
            
            # Add entities and extract relationships via dependency parsing
            self.graph_builder.add_chunk(idx, entities, chunk)

            # Embed chunk for similarity search (used in Equations 4 & 5)
            chunk_embeddings[idx] = self.embedder.encode(chunk)
            
            # Embed entities for local search (Equation 4)
            for entity in entities:
                if entity not in entity_embeddings:
                    entity_embeddings[entity] = self.embedder.encode(entity)

        # Step 4: Get constructed knowledge graph
        graph = self.graph_builder.get_graph()
        
        # Step 5: Community Detection (Louvain algorithm)
        communities = CommunityDetector().detect(graph)

        # Step 6: Generate community summaries for global search
        summaries = {
            cid: CommunitySummarizer().summarize(entities)
            for cid, entities in communities.items()
        }
        
        # Step 7: Save outputs to data/processed/
        self._save_outputs(chunks, graph, chunk_embeddings, entity_embeddings, 
                          communities, summaries)

        return chunks, graph, chunk_embeddings, entity_embeddings, summaries
    
    def _save_outputs(self, chunks, graph, chunk_embeddings, entity_embeddings,
                     communities, summaries):
        """
        Save pipeline outputs to data/processed/.
        
        Saves:
        - chunks.json: List of semantic chunks with metadata
        - knowledge_graph.pkl: Complete graph with embeddings and communities
        
        Args:
            chunks: List of text chunks
            graph: NetworkX graph object
            chunk_embeddings: Dict mapping chunk_id -> embedding
            entity_embeddings: Dict mapping entity -> embedding
            communities: Dict mapping community_id -> list of entities
            summaries: Dict mapping community_id -> summary text
        """
        # Save chunks.json
        chunks_data = {
            "chunks": [
                {
                    "id": idx,
                    "text": chunk,
                    "length": len(chunk)
                }
                for idx, chunk in enumerate(chunks)
            ],
            "total_chunks": len(chunks)
        }
        
        chunks_path = self.output_dir / "chunks.json"
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        # Save knowledge_graph.pkl
        graph_data = {
            "graph": graph,
            "chunk_embeddings": chunk_embeddings,
            "entity_embeddings": entity_embeddings,
            "communities": communities,
            "community_summaries": summaries,
            "metadata": {
                "num_entities": graph.number_of_nodes(),
                "num_relationships": graph.number_of_edges(),
                "num_communities": len(communities),
                "num_chunks": len(chunks)
            }
        }
        
        graph_path = self.output_dir / "knowledge_graph.pkl"
        with open(graph_path, 'wb') as f:
            pickle.dump(graph_data, f)
        
        print(f"\n✓ Saved outputs to {self.output_dir}/")
        print(f"  - chunks.json ({len(chunks)} chunks)")
        print(f"  - knowledge_graph.pkl ({graph.number_of_nodes()} entities, "
              f"{graph.number_of_edges()} relationships, {len(communities)} communities)")
