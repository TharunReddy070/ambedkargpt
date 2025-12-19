"""
SEMRAG: Semantic Graph-based Retrieval-Augmented Generation System

Main entry point for the SEMRAG pipeline as per the research paper.
Implements semantic chunking, knowledge graph construction, community detection,
and dual-mode (local/global) retrieval for question answering.
"""
import argparse
import yaml

from src.pipeline.ambedkargpt import load_pdf_text, SemRAGPipeline
from src.retrieval.local_search import LocalGraphSearch
from src.retrieval.global_search import GlobalGraphSearch
from src.retrieval.ranker import HybridRanker
from src.llm.prompt_templates import PromptBuilder
from src.llm.answer_generator import AnswerGenerator
from src.llm.llm_client import OllamaClient


def load_config():
    """Load configuration from config.yaml."""
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)


def decide_mode(query: str, entities: list[str]) -> str:
    """
    Auto-select retrieval mode based on query characteristics.
    
    Local mode: For entity-specific, narrow questions
    Global mode: For broad, conceptual questions
    """
    query_words = query.split()
    is_short_query = len(query_words) <= 12
    has_entity_mention = any(e.lower() in query.lower() for e in entities)
    
    if is_short_query and has_entity_mention:
        return "local"
    return "global"


def main():
    parser = argparse.ArgumentParser(
        description="SEMRAG: Semantic Graph-based RAG for Question Answering"
    )
    parser.add_argument("--file", required=True, help="Path to input PDF file")
    parser.add_argument("--question", required=True, help="Question to answer")
    parser.add_argument(
        "--mode",
        choices=["local", "global", "auto"],
        default="auto",
        help="Retrieval mode: local (entity-focused), global (topic-focused), or auto"
    )
    args = parser.parse_args()

    print("\n" + "="*70)
    print("SEMRAG: Semantic Graph-based RAG System")
    print("="*70)

    # Stage 1: PDF Text Extraction
    print("\n[Stage 1] Extracting text from PDF...")
    text = load_pdf_text(args.file)
    print(f"✓ Extracted {len(text)} characters")

    # Stage 2: Build Knowledge Graph (Semantic Chunking + Entity Extraction + Communities)
    print("\n[Stage 2] Building SEMRAG Knowledge Graph...")
    print("  - Semantic chunking (Algorithm 1)")
    print("  - Entity extraction")
    print("  - Graph construction")
    print("  - Community detection")
    print("  - Saving to data/processed/")
    
    pipeline = SemRAGPipeline()
    chunks, graph, chunk_embs, entity_embs, summaries = pipeline.build(text)
    
    print(f"✓ Created {len(chunks)} semantic chunks")
    print(f"✓ Extracted {graph.number_of_nodes()} entities")
    print(f"✓ Built graph with {graph.number_of_edges()} relationships")
    print(f"✓ Detected {len(summaries)} communities")

    # Stage 3: Mode Selection
    mode = args.mode
    if mode == "auto":
        entities_list = list(graph.nodes())
        mode = decide_mode(args.question, entities_list)
        print(f"\n[Stage 3] Auto-selected mode: {mode}")
    else:
        print(f"\n[Stage 3] Using specified mode: {mode}")

    # Stage 4: Retrieval with Hybrid Re-ranking
    print("\n[Stage 4] Retrieving relevant context...")
    
    # Load retrieval config
    config = load_config()
    retrieval_config = config.get('retrieval', {})
    hybrid_config = retrieval_config.get('hybrid', {})
    
    # Local Graph Retrieval (Equation 4) with scores
    print("  - Local search (entity-focused, Equation 4)")
    local_retriever = LocalGraphSearch(pipeline.embedder)
    local_results = local_retriever.search_with_scores(args.question, entity_embs, chunk_embs)
    print(f"    Retrieved {len(local_results)} relevant chunks")

    # Global Graph Retrieval (Equation 5) with scores
    print("  - Global search (community-focused, Equation 5)")
    global_retriever = GlobalGraphSearch(pipeline.embedder)
    global_results = global_retriever.search_with_scores(args.question, summaries)
    print(f"    Retrieved {len(global_results)} relevant communities")
    
    # Hybrid Re-ranking: Combine local + global results
    print("  - Hybrid re-ranking (combining local + global)")
    ranker = HybridRanker()  # Loads alpha from config
    
    # Rank combines local chunk scores with global community scores
    # Returns unified ranking of chunk IDs
    ranked_chunk_ids = ranker.rank(local_results, global_results)
    
    # Take top-K after re-ranking (from config)
    top_k_final = hybrid_config.get('final_top_k', 5)
    final_chunk_ids = ranked_chunk_ids[:top_k_final]
    retrieved_local_chunks = [chunks[i] for i in final_chunk_ids]
    
    # Get communities for context
    retrieved_community_ids = [cid for cid, _ in global_results[:3]]
    retrieved_summaries = [summaries[cid] for cid in retrieved_community_ids]
    
    print(f"    Final re-ranked results: {top_k_final} chunks + {len(retrieved_community_ids)} communities")

    # Stage 5: Answer Generation with Citations
    print("\n[Stage 5] Generating answer with source citations...")
    prompt_builder = PromptBuilder()
    prompt = prompt_builder.build(
        args.question, 
        retrieved_local_chunks, 
        retrieved_summaries,
        final_chunk_ids,
        retrieved_community_ids
    )

    llm = OllamaClient()
    generator = AnswerGenerator(llm)
    answer = generator.generate(prompt)

    print("\n" + "="*70)
    print("ANSWER:")
    print("="*70)
    print(answer)
    print("="*70)
    print("\nSOURCES (After Hybrid Re-ranking):")
    print(f"  Chunks: {final_chunk_ids}")
    print(f"  Communities: {retrieved_community_ids}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
