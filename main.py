import argparse
from llm import ollama_client
from pypdf import PdfReader

from src.llm.ollama_client import OllamaClient
from src.pipeline.semrag_pipeline import SemRAGPipeline
from src.retrieval.local_search import LocalGraphSearch
from src.retrieval.global_search import GlobalGraphSearch
from src.llm.prompt_templates import PromptBuilder
from src.llm.answer_generator import AnswerGenerator


def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() for page in reader.pages)


def decide_mode(query: str, entities: list[str]) -> str:
    if len(query.split()) <= 12 and any(e.lower() in query.lower() for e in entities):
        return "local"
    return "global"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--mode", choices=["local", "global", "auto"], default="auto")
    args = parser.parse_args()

    text = load_pdf(args.file)
    pipeline = SemRAGPipeline()

    chunks, graph, chunk_embs, entity_embs, summaries = pipeline.build(text)

    mode = args.mode
    if mode == "auto":
        mode = decide_mode(args.question, list(graph.nodes))

    if mode == "local":
        retriever = LocalGraphSearch(pipeline.embedder)
        ids = retriever.search(args.question, entity_embs, chunk_embs)
        local_ctx = "\n".join(chunks[i] for i in ids)
        global_ctx = ""
    else:
        retriever = GlobalGraphSearch(pipeline.embedder)
        cids = retriever.search(args.question, summaries)
        global_ctx = "\n".join(summaries[cid] for cid in cids)
        local_ctx = ""

    prompt = PromptBuilder().build(args.question, local_ctx, global_ctx)

    ollama_client = OllamaClient(model="llama3.2:latest", temperature=0.2)
    answer = AnswerGenerator(ollama_client).generate(prompt)


    print(answer)


if __name__ == "__main__":
    main()
