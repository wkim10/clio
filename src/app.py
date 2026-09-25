from typing import Literal
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import json

# shared RAG pieces live in query.py -> the ChromaDB and Anthropic clients are created once there
# and reused across all requests (importing query.py also loads the .env file)
from src.query import anthropic_client, retrieve, build_system_prompt, MODEL, MAX_TOKENS

# initialize FastAPI app -> FastAPI handles routing, request parsing, and response serialization
app = FastAPI()

# serve files from the static/ folder at the /static URL path
# allows index.html to load CSS, JS, and other assets
app.mount("/static", StaticFiles(directory="static"), name="static")

# cap how much conversation history is sent to Claude and back
# -> keeps prompts (and cost) from growing without limit in long conversations
MAX_HISTORY_MESSAGES = 20  # 10 question/answer exchanges

# pydantic models to define the shape of request and response JSON
# FastAPI will automatically validate incoming requests against these models

class Message(BaseModel):
    """A single message in the conversation - either from the user or Clio."""
    # only roles the Claude API accepts in messages -> anything else is rejected with a 422
    role: Literal["user", "assistant"]
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

def trim_history(history):
    """Keep only the most recent messages, making sure the conversation still starts with a user message."""
    trimmed = history[-MAX_HISTORY_MESSAGES:]
    while trimmed and trimmed[0].role != "user":
        trimmed = trimmed[1:]
    return trimmed

def build_search_query(question, history):
    """
    Combine the last user message from history with the current question
    so retrieval is grounded in the conversation context
    (e.g. "Who founded it?" after a question about the Tang dynasty).
    """
    last_user_message = next(
        (m.content for m in reversed(history) if m.role == "user"),
        ""
    )
    return f"{last_user_message} {question}".strip()

def prepare(request):
    """
    Steps shared by both endpoints, before generation.

    1. Trim the conversation history
    2. Build a context-aware search query using conversation history
    3. Retrieve labeled chunks from ChromaDB
    4. Build the system prompt and the messages array for Claude

    Returns (history, sources, system_prompt, messages).
    """
    history = trim_history(request.history)
    context, sources = retrieve(build_search_query(request.question, history))

    # the system prompt holds the instructions and retrieved context,
    # so the messages array stays a clean record of the conversation
    system_prompt = build_system_prompt(context)

    # passing the history gives Claude memory of the conversation, then append the new question
    messages = [m.model_dump() for m in history]
    messages.append({"role": "user", "content": request.question})

    return history, sources, system_prompt, messages

def build_updated_history(history, question, answer):
    """Append the new exchange -> returned to the frontend and sent back on the next request."""
    # model_dump() converts Message objects to dicts -> json.dumps can't serialize pydantic models
    return [m.model_dump() for m in history] + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ]

@app.get("/")
def root():
    """Serve the main HTML page when the user visits the root URL"""
    return FileResponse("static/index.html")

@app.post("/query")
def query(request: QueryRequest):
    """
    Non-streaming RAG endpoint -> returns the full answer in one response.
    The web UI uses /stream; this is handy for scripts and curl.
    """
    history, sources, system_prompt, messages = prepare(request)

    # call Claude Haiku with the system prompt and conversation history
    response = anthropic_client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=messages
    )

    answer = response.content[0].text

    # return the answer, sources, and updated history to the frontend
    return QueryResponse(
        answer=answer,
        sources=sources,
        history=build_updated_history(history, request.question, answer)
    )

# a plain def (not async def) so FastAPI runs it in a worker thread
# -> the blocking ChromaDB query doesn't stall other requests
@app.post("/stream")
def stream(request: QueryRequest):
    """
    Streaming RAG endpoint -> sends response tokens as they are generated
    instead of waiting for the full response.
    """
    history, sources, system_prompt, messages = prepare(request)

    # generator function that yields tokens as they arrive from Claude
    def generate():
        full_answer = ""

        with anthropic_client.messages.stream(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=messages
        ) as stream:
            for text in stream.text_stream:
                full_answer += text
                yield f"data: {json.dumps({'type': 'token', 'text': text})}\n\n"

        # after streaming completes, send sources and updated history
        updated_history = build_updated_history(history, request.question, full_answer)
        yield f"data: {json.dumps({'type': 'done', 'sources': sources, 'history': updated_history})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
