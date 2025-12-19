class AnswerGenerator:
    """
    Generate answers using LLM with source citations.
    
    Implements the final step of SEMRAG pipeline:
    - Combines retrieved local entities and global community summaries
    - Creates prompt with context, entities, and query
    - Uses local LLM (Llama3/Mistral) to generate answer
    - Includes citations to source chunks
    """
    
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def generate(self, prompt: str) -> str:
        """
        Generate answer from prompt.
        
        Args:
            prompt: Formatted prompt with context and citations
            
        Returns:
            Generated answer with source citations
        """
        return self.llm_client.generate(prompt)
