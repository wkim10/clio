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

class Message(BaseModel):
    """A single message in the conversation - either from the user or Clio."""
    role: str
    content: str

class QueryRequest(BaseModel):
    """What the frontend sends -> a single question string"""
    question: str
    history: list[Message] = []

class QueryResponse(BaseModel):
    """What we send back -> the generated answer and list of source file paths"""
    answer: str
    sources: list[str]
    history: list[Message]

@app.get("/")
def root():
    """Serve the main HTML page when the user visits the root URL"""
    return FileResponse("static/index.html")

@app.post("/query")
def query(request: QueryRequest):
    """
    Main RAG endpoint -> the full retrieval and generation pipeline in one request.

    Flow:
    1. Receive question and conversation history from frontend
    2. Build a context-aware search query using conversation history
    3. Query ChromaDB for the most semantically similar chunks
    4. Build a system prompt combining Clio's instructions and retrieved context
    5. Build the messages array from conversation history + new question
    6. Send to Claude to generate a grounded, history-aware answer
    7. Append new exchange to history and return everything to the frontend
    """

    # step 1 -> BUILD SEARCH QUERY
    # combine the last user message from history with the current question
    # so retrieval is grounded in the conversation context
    if request.history:
        last_user_message = next(
            (m.content for m in reversed(request.history) if m.role == "user"),
            ""
        )
        search_query = f"{last_user_message} {request.question}"
    else:
        search_query = request.question

    # step 2 -> RETRIEVE
    # ChromaDB converts the question to an embedding and finds
    # the 10 most similar chunks using cosine similarity
    results = collection.query(
        query_texts=[search_query],
        n_results=10
    )
    
    # join retrieved chunks into a single context string for the system prompt
    context = "\n\n".join(results["documents"][0])

    # deduplicate source file paths -> multiple chunks may come from the same file
    sources = list(set(r["source"] for r in results["metadatas"][0]))

    # step 3 -> BUILD SYSTEM PROMPT
    # the system prompt contains Clio's instructions and the retrieved context
    # it's separate from the messages array so conversation history stays clean
    system_prompt = f"""You are Clio, a personal history research assistant.
Answer questions based on the context provided from the user's personal history notes and documents.
If the answer is not in the context, say so honestly.

Context from knowledge base:
{context}"""
    
    # step 4 -> BUILD MESSAGES ARRAY
    # convert history to the format Claude expects, then append the new question
    # passing full history gives Claude memory of the entire conversation
    messages = [{"role": m.role, "content": m.content} for m in request.history]
    messages.append({"role": "user", "content": request.question})
    
    # step 5 -> GENERATE
    # call Claude Haiku with the system prompt and full conversation history
    response = anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        system=system_prompt,
        messages=messages
    )

    answer = response.content[0].text

    # step 6 -> UPDATE HISTORY
    # append the new user question and Clio's answer to the history
    # this updated history is returned to the frontend and sent back on the next request
    updated_history = list(request.history) + [
        Message(role="user", content=request.question),
        Message(role="assistant", content=answer)
    ]

    # return the answer, sources, and updated history to the frontend
    return QueryResponse(
        answer=answer,
        sources=sources,
        history=updated_history
    )