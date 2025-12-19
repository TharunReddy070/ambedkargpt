# SEMRAG: Semantic Graph-based Retrieval-Augmented Generation

Implementation of the SEMRAG research paper for question-answering over Dr. B.R. Ambedkar's writings using semantic chunking, knowledge graph construction, and hybrid retrieval with citation tracking.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PDF Document Input                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 1: Text Extraction                      │
│                        (pypdf library)                           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              Stage 2: Semantic Chunking (Algorithm 1)            │
│  • Sentence tokenization (NLTK)                                  │
│  • Buffer merging for context                                    │
│  • Cosine similarity-based grouping                              │
│  • Token limit enforcement (1024 max, 128 sub-chunks)            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Stage 3: Knowledge Graph Construction            │
│  • Entity extraction (spaCy NER)                                 │
│  • Relationship extraction (Dependency parsing)                  │
│  • Graph building (NetworkX)                                     │
│  • Community detection (Louvain algorithm)                       │
│  • Community summarization (LLM)                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 4: Hybrid Retrieval                     │
│  ┌─────────────────────┐      ┌────────────────────────┐        │
│  │  Local Search (Eq 4) │      │ Global Search (Eq 5)   │        │
│  │  Entity-focused      │      │ Community-focused      │        │
│  │  Returns: Chunks     │      │ Returns: Communities   │        │
│  └──────────┬───────────┘      └───────────┬────────────┘        │
│             │                               │                     │
│             └───────────┬───────────────────┘                     │
│                         ▼                                         │
│              ┌───────────────────────┐                            │
│              │  Hybrid Re-ranker     │                            │
│              │  (Weighted fusion)    │                            │
│              └──────────┬────────────┘                            │
│                         │                                         │
│                         ▼                                         │
│              Top-K Chunks + Communities                           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│            Stage 5: Answer Generation with Citations             │
│  • Context building with [Chunk-ID] and [Community-ID] labels    │
│  • LLM generation (Ollama llama3.2)                              │
│  • Source citation extraction                                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Citation-backed Answer                        │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
SEMRAG/
├── data/
│   ├── Ambedkar_works.pdf          # Source document
│   └── processed/                   # Generated artifacts
│       ├── chunks.json
│       └── knowledge_graph.pkl
├── src/
│   ├── chunking/                    # Semantic chunking
│   │   ├── semantic_chunker.py     # Algorithm 1 implementation
│   │   └── buffer_merger.py
│   ├── graph/                       # Knowledge graph
│   │   ├── entity_extractor.py
│   │   ├── graph_builder.py
│   │   ├── community_detector.py
│   │   └── summarizer.py
│   ├── retrieval/                   # Search modules
│   │   ├── local_search.py         # Equation 4
│   │   ├── global_search.py        # Equation 5
│   │   └── ranker.py               # Hybrid re-ranking
│   ├── llm/                         # LLM integration
│   │   ├── llm_client.py
│   │   ├── prompt_templates.py
│   │   └── answer_generator.py
│   └── pipeline/
│       └── ambedkargpt.py          # Main pipeline
├── tests/
├── config.yaml                      # Configuration
├── requirements.txt
├── main.py
└── README.md
```

## Installation

### Prerequisites

- Python 3.11 or higher
- Ollama (for local LLM inference)
- 8GB RAM minimum (16GB recommended for large documents)

### Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/TharunReddy070/SEMRAG.git
cd SEMRAG
```

2. **Create and activate virtual environment:**

Windows:
```bash
python -m venv venv311
.\venv311\Scripts\activate
```

Linux/Mac:
```bash
python -m venv venv311
source venv311/bin/activate
```

3. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download required NLP models:**
```bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

5. **Install and configure Ollama:**

Download from [https://ollama.com](https://ollama.com)

Pull the required model:
```bash
ollama pull llama3.2
```

6. **Verify installation:**
```bash
python main.py --file data/Ambedkar_works.pdf --question "Who was Dr. Ambedkar?" --mode auto
```

## Usage

### Command Line Interface

Run the system with the following command:

```bash
python main.py --file data/Ambedkar_works.pdf --question "What were Ambedkar's views on caste system?" --mode auto
```

**Arguments:**
- `--file`: Path to PDF document (required)
- `--question`: Question to answer (required)
- `--mode`: Retrieval strategy - `local`, `global`, or `auto` (default: auto)

**Retrieval Modes:**
- `local`: Entity-focused search for specific queries
- `global`: Community-based search for broad questions
- `auto`: Automatic mode selection based on query analysis

### Example Output

```
======================================================================
SEMRAG: Semantic Graph-based RAG System
======================================================================

[Stage 1] Extracting text from PDF...
Extracted 266194 characters

[Stage 2] Building SEMRAG Knowledge Graph...
  - Semantic chunking (Algorithm 1)
  - Entity extraction
  - Graph construction
  - Community detection
  - Saving to data/processed/

Saved outputs to data/processed/
  - chunks.json (103 chunks)
  - knowledge_graph.pkl (690 entities, 1209 relationships, 16 communities)

[Stage 3] Auto-selected mode: local

[Stage 4] Retrieving relevant context...
  - Local search (entity-focused, Equation 4)
    Retrieved 10 relevant chunks
  - Global search (community-focused, Equation 5)
    Retrieved 5 relevant communities
  - Hybrid re-ranking (combining local + global)
    Final re-ranked results: 5 chunks + 3 communities

[Stage 5] Generating answer with source citations...

======================================================================
ANSWER:
======================================================================
Ambedkar's views on caste system are as follows:

1. Caste is an "unnatural institution": Ambedkar described caste as a 
   mechanism for endogamy, which he believed was the origin of the caste 
   system in India.
2. Brahmins are the originators of caste: He argued that the Brahmin class 
   is responsible for creating and maintaining the caste system.
3. Caste was not imposed by a law-giver: Ambedkar rejected the idea that 
   the caste system was imposed by a law-giver such as Manu.
4. Caste is a product of religious beliefs: He believed that caste emerged 
   from certain religious beliefs and practices sanctioned by Shastras.
5. Destruction of caste requires destroying its sacredness: Ambedkar 
   emphasized that destroying the caste system would require destroying 
   the authority of the Shastras and Vedas.

References: [Chunk-6], [Chunk-66]

SOURCES (After Hybrid Re-ranking):
  Chunks: [73, 42, 7, 6, 66]
  Communities: [7, 6, 5]
======================================================================
```

## Configuration

All parameters are configurable via `config.yaml`:

```yaml
chunking:
  similarity_threshold: 0.3    # Cosine distance threshold
  buffer_size: 2               # Context window size
  max_tokens: 1024             # Maximum chunk size
  subchunk_tokens: 128         # Sub-chunk target size
  overlap_tokens: 20           # Overlap between sub-chunks

retrieval:
  local:
    tau_e: 0.3                 # Entity similarity threshold
    tau_d: 0.3                 # Chunk relevance threshold
    top_k: 10                  # Number of chunks to retrieve
  global:
    top_k: 5                   # Number of communities to retrieve
  hybrid:
    alpha: 0.6                 # Local weight (1-alpha for global)
    final_top_k: 5             # Final chunks after re-ranking

llm:
  model: llama3.2              # Ollama model name
  temperature: 0.7             # Generation temperature

embeddings:
  model: all-MiniLM-L6-v2      # Sentence transformer model
  dimension: 384               # Embedding dimension
```

## License

MIT License

## Author

Tharun Reddy - [GitHub](https://github.com/TharunReddy070)
