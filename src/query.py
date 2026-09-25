import chromadb
import anthropic
from pathlib import Path
from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()

# settings shared by the CLI below and the web app (app.py imports these)
MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 1000
N_RESULTS = 10

# connect to the existing ChromaDB database on disk
# this must already exist -> run ingest.py first to create it
client = chromadb.PersistentClient(path="./chroma_db")

# get the "clio" collection -> raises an error if it doesn't exist
collection = client.get_collection("clio")

# initialize the Anthropic client
anthropic_client = anthropic.Anthropic()

def retrieve(search_query, n_results=N_RESULTS):
    """
    RETRIEVE step of RAG -> find the chunks most similar to the search query
    and label each one with a source number the model can cite.

    Returns (context, sources):
    - context: the chunks joined into one string, each starting with a label like "[2] Lecture 17.pdf"
    - sources: unique source file paths in relevance order -> sources[0] is [1], sources[1] is [2], ...
    """

    # ChromaDB converts the query to an embedding and finds the
    # n_results nearest chunks (L2 distance by default)
    results = collection.query(
        query_texts=[search_query],
        n_results=n_results
    )

    documents = results["documents"][0]
    chunk_sources = [m["source"] for m in results["metadatas"][0]]

    # deduplicate while keeping relevance order -> set() would scramble it
    sources = list(dict.fromkeys(chunk_sources))

    # chunks from the same file share a number so citations point to files, not chunks
    labeled_chunks = [
        f"[{sources.index(source) + 1}] {Path(source).name}\n{document}"
        for document, source in zip(documents, chunk_sources)
    ]

    return "\n\n".join(labeled_chunks), sources

def build_system_prompt(context):
    """Combine Clio's instructions with the labeled context from retrieve()."""
    return f"""You are Clio, a personal history research assistant.
Answer questions based on the context provided from the user's personal history notes and documents.
If the answer is not in the context, say so honestly.

Each excerpt in the context starts with a source number in brackets, like [1].
Cite the sources you draw from inline using those numbers, e.g. "The Tang dynasty was founded in 618 [2]."
Only cite numbers that appear in the context, and don't add a separate list of sources at the end.

Context from knowledge base:
{context}"""

def query(question):
    """
    Query Clio with a history question using RAG (Retrieval Augmented Generation)

    The RAG pipeline has two steps:
    1. RETRIEVE: find the most semantically similar chunks from ChromaDB
    2. GENERATE: pass those chunks as context to Claude to generate an answer
    """

    # step 1 - RETRIEVE
    context, sources = retrieve(question)

    # step 2 - GENERATE
    # the instructions and labeled context go in the system prompt, the question in the user message
    response = anthropic_client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=build_system_prompt(context),
        messages=[{"role": "user", "content": question}]
    )

    # response.content is a list of content blocks
    # [0].text extracts the text from the first (and only) block
    print("\n--- Answer ---")
    print(response.content[0].text)

    # print sources numbered to match the citations in the answer
    print("\n--- Sources ---")
    for i, source in enumerate(sources, start=1):
        print(f"  [{i}] {source}")

if __name__ == "__main__":
    # simple interactive loop -> keep asking questions until user types quit/exit
    while True:
        question = input("\nAsk Clio: ")
        if question.lower() in ["quit", "exit"]:
            break
        query(question)
