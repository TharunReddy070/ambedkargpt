class AnswerGenerator:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def generate(self, prompt: str) -> str:
        return self.llm_client.generate(prompt)
