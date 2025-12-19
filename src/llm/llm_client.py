import yaml
from pathlib import Path
import ollama


def load_config():
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class OllamaClient:
    def __init__(self, model: str = None, temperature: float = None):
        # Load from config if not provided
        config = load_config()
        llm_config = config.get('llm', {})
        
        self.model = model or llm_config.get('model', 'llama3.2')
        self.temperature = temperature if temperature is not None else llm_config.get('temperature', 0.7)

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
