import chromadb
import anthropic
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()

# initialize FastAPI app -> FastAPI handles routing, request parsing, and response serialization
app = FastAPI()

# initialize ChromaDB and Anthropic clients
# created once at startup and reused across all requests
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("clio")
anthropic_client = anthropic.Anthropic()

# serve files from the static/ folder at the /static URL path
# allows index.html to load CSS, JS, and other assets
app.mount("/static", StaticFiles(directory="static"), name="static")

# pydantic models to define the shape of request and response JSON
# FastAPI will automatically validate incoming requests against these models

class QueryRequest(BaseModel):
    """What the frontend sends -> a single question string"""
    question: str

class QueryResponse(BaseModel):
    """What we send back -> the generated answer and list of source file paths"""
    answer: str
    sources: list[str]

@app.get("/")
def root():
    """Serve the main HTML page when the user visits the root URL"""
    return FileResponse("static/index.html")

@app.post("/query")
def query(request: QueryRequest):
    """
    Main RAG endpoint -> the full retrieval and generation pipeline in one request.

    Flow:
    1. Receive question from frontend
    2. Query ChromaDB for the most semantically similar chunks
    3. Build a prompt combining the retrieved context and the question
    4. Send to Claude to generate a grounded answer
    5. Return the answer and source file paths to the frontend
    """

    # step 1 -> RETRIEVE
    # ChromaDB converts the question to an embedding and finds
    # the 5 most similar chunks using cosine similarity
    results = collection.query(
        query_texts=[request.question],
        n_results=5
    )
    
    # join retrieved chunks into a single context string for the prompt
    context = "\n\n".join(results["documents"][0])

    # deduplicate source file paths -> multiple chunks may come from the same file
    sources = list(set(r["source"] for r in results["metadatas"][0]))

    # step 2 -> GENERATE
    # instruct Claude to answer using only the retrieved context
    prompt = f"""You are Clio, a personal history research assistant.
Answer the question based on the context provided from the user's personal history notes and documents.

Context:
{context}

Question: {request.question}

Answer thoughtfully and cite which sources you drew from."""
    
    # call Claude Haiku
    response = anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    # return the answer text and source list to the frontend
    return QueryResponse(
        answer=response.content[0].text,
        sources=sources
    )