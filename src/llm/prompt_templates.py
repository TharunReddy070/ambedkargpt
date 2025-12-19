class PromptBuilder:
    def build(
        self,
        question: str,
        local_context: str,
        global_context: str,
    ) -> str:
        return f"""
Context (Global):
{global_context}

Context (Local):
{local_context}

Question:
{question}

Answer based only on the above context.
"""
