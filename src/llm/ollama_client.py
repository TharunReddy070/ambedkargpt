import ollama


class OllamaClient:
    def __init__(self, model: str = "llama3.2:latest", temperature: float = 0.2):
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": "Answer strictly using the provided context."},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": self.temperature},
        )
        return response["message"]["content"]
