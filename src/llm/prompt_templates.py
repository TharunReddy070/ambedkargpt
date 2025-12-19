class PromptBuilder:
    """Build prompts for LLM with context and citations."""
    
    def build(
        self,
        question: str,
        local_chunks: list,
        global_summaries: list,
        local_chunk_ids: list = None,
        community_ids: list = None
    ) -> str:
        """
        Build prompt with citations.
        
        Args:
            question: User question
            local_chunks: List of relevant chunk texts
            global_summaries: List of community summary texts
            local_chunk_ids: List of chunk IDs for citation
            community_ids: List of community IDs for citation
            
        Returns:
            Formatted prompt with numbered sources
        """
        # Build context with citations
        context_parts = []
        
        # Add global context with citations
        if global_summaries:
            context_parts.append("\n=== Community Summaries (Broad Context) ===")
            for idx, (cid, summary) in enumerate(zip(community_ids or range(len(global_summaries)), global_summaries), 1):
                context_parts.append(f"[Community-{cid}] {summary}")
        
        # Add local context with citations
        if local_chunks:
            context_parts.append("\n=== Relevant Chunks (Specific Context) ===")
            for idx, (chunk_id, chunk) in enumerate(zip(local_chunk_ids or range(len(local_chunks)), local_chunks), 1):
                context_parts.append(f"[Chunk-{chunk_id}] {chunk}")
        
        context = "\n\n".join(context_parts)
        
        return f"""
You are an expert assistant analyzing Dr. B.R. Ambedkar's works.

{context}

Question: {question}

Instructions:
1. Answer based ONLY on the context provided above.
2. Include citations using the format [Chunk-ID] or [Community-ID] to reference your sources.
3. If the context doesn't contain enough information, state that clearly.
4. Be comprehensive but concise.

Answer:
"""
