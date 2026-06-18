import chromadb
import anthropic
from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()

# connect to the existing ChromaDB database on disk
# this must already exist -> run ingest.py first to create it
client = chromadb.PersistentClient(path="./chroma_db")

# get the "clio" collection -> raises an error if it doesn't exist
collection = client.get_collection("clio")

# initialize the Anthropic client
anthropic_client = anthropic.Anthropic()

def query(question, n_results=5):
    """
    Query Clio with a history question using RAG (Retrieval Augmented Generation)

    The RAG pipeline has two steps:
    1. RETRIEVE: find the most semantically similar chunks from ChromaDB
    2. GENERATE: pass those chunks as context to Claude to generate an answer

    n_results: number of document chunks to retrieve and pass as context
    """

    # step 1 - RETRIEVE
    # ChromaDB converts the question to an embedding and finds the
    # n_results most similar chunks using cosine similarity
    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )

    # results["documents"][0] is a list of the retireved text chunks
    # join them together with blank lines as the context for the LLM
    context = "\n\n".join(results["documents"][0])

    # extract the source file path from each chunk's metadata
    sources = [r["source"] for r in results["metadatas"][0]]

    # step 2 - GENERATE
    # build a prompt that instructs Claude to answer using only the
    # retrieved context, grounding the response in my actual notes
    prompt = f"""You are Clio, a personal history research assistant. 
Answer the question based on the context provided from the user's personal history notes and documents.

Context:
{context}

Question: {question}

Answer thoughtfully and cite which sources you drew from."""
    
    # call the Anthropic API with Claude Haiku
    # the retrieved context + question are passed as the user message
    response = anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    # response.content is a list of content blocks
    # [0].text extracts the text from the first (and only) block
    print("\n--- Answer ---")
    print(response.content[0].text)

    # deduplicate sources with set() since multiple chunks
    # may come from the same file
    print("\n--- Sources ---")
    for source in set(sources):
        print(f"  {source}")

if __name__ == "__main__":
    # simple interactive loop -> keep asking questions until user types quit/exit
    while True:
        question = input("\nAsk Clio: ")
        if question.lower() in ["quit", "exit"]:
            break
        query(question)