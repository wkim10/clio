# Clio

> Named after the Greek muse of history.

Clio is a personal history knowledge base and research assistant powered by RAG (Retrieval Augmented Generation). It ingests your own history notes, essays, and documents into a vector database and lets you query them using natural language -> surfacing your own knowledge and research in response to questions.

## What It Does

Ask Clio questions about history and it answers using **your own notes and documents** as the source of truth, not generic internet knowledge. It retrieves the most relevant passages from your personal knowledge base and passes them to Claude to generate a grounded, cited response.

**Example queries:**

- "What were the main causes of the fall of the Roman Republic?"
- "What similarities exist between the Han Dynasty and the Roman Empire?"
- "What did I write about tea culture in Tang Dynasty China?"
- "Who was Justinian and what were his major accomplishments?"

## Data Sources

Clio is built on my personal history knowledge base accumulated over several years, spanning:

- **College coursework** across eight history courses:
  - ARCH-0128: Mesoamerican Archaeology
  - CLS-0038: History of Ancient Rome
  - HIST-0040: History of Pre-Modern China
  - HIST-0041: Modern Chinese History
  - HIST-0053: Europe to 1815
  - HIST-0054: Europe Since 1815
  - HIST-0058: The Byzantines and Their World
  - HIST-0089: What is History?
- **Personal research notes** on topics including Japanese history by period, Korean history, US presidential history, French history, and ancient Rome
- **Personal essays and papers** written across multiple history courses
- **Kindle highlights** from history books in my personal library

~5,670 chunks ingested across 500+ documents.

## How It Works

```
Personal notes + essays + course documents
             ↓
    Text extraction (docx, pdf)
             ↓
    Chunking (500 words, 50 word overlap)
             ↓
    Embeddings via ChromaDB (all-MiniLM-L6-v2)
             ↓
    Stored in persistent ChromaDB vector database
             ↓
    Query → retrieve top 5 relevant chunks → pass to Claude
             ↓
    Grounded answer with source citations
```

## Tech Stack

- **ChromaDB** — local persistent vector database with built-in embeddings
- **Anthropic API (Claude Haiku)** — LLM for answer generation
- **pypdf** — PDF text extraction
- **docx2txt** — Word document text extraction
- **python-dotenv** — environment variable management

## Setup

To run Clio with your own documents, replace the contents of the `data/` folder with your own notes and documents, then run the ingestion pipeline.

Prerequisites: Python 3.11 or earlier (Python 3.13 has torch compatibility issues)

```bash
# clone the repo
git clone https://github.com/wkim10/clio.git
cd clio

# create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install chromadb anthropic python-dotenv pypdf docx2txt
```

Add your Anthropic API key:

```bash
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

Add your documents to the data/ folder:

```
clio/
└── data/
    ├── google_docs/
    ├── notion/
    └── kindle/
```

Ingest your documents:

```bash
python src/ingest.py
```

Query Clio:

```bash
python src/query.py
```

## Project Structure

```
clio/
├── src/
│   ├── ingest.py    # document ingestion pipeline
│   └── query.py     # RAG query interface
├── data/            # personal documents (gitignored)
├── chroma_db/       # vector database (gitignored)
├── .env             # API keys (gitignored)
└── README.md
```

## Limitations

- Data is personal and not included in the repository
- Requires your own Anthropic API key
- Best results with your own notes rather than downloaded textbooks
- Answer quality depends on what is in your personal knowledge base
