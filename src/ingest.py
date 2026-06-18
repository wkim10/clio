import os
import chromadb
import docx2txt
import pypdf
from pathlib import Path
from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()

# initialize a persistent ChromaDB client -> stores the vector database on disk
# so embeddings are saved between runs and don't need to be recomputed
client = chromadb.PersistentClient(path="./chroma_db")

# get or create a collection called "clio"
collection = client.get_or_create_collection("clio")

def extract_text_from_docx(path):
    """Extract raw text from a .docx Word document."""
    try:
        return docx2txt.process(path)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return ""

def extract_text_from_pdf(path):
    """Extract raw text from a PDF by reading each page individually."""
    try:
        reader = pypdf.PdfReader(path)
        # extract text from each page, defaulting to "" if a page has no text
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return ""

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into overlapping chunks of words.

    chunk_size: number of words per chunk
    overlap: number of words shared between consecutive chunks

    Overlap ensures that sentences spanning chunk boundaries aren't
    lost -> context carries over from one chunk to the next
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def ingest_file(path):
    """
    Extract text from a single file, chunk it, and store in ChromaDB.

    Each chunk is stored as a separate document with metadata tracking
    its source file and position. ChromaDB automatically generates
    embeddings for each chunk using its built-in embedding model.

    Returns the number of chunks ingested (0 if file was skipped).
    """
    path = Path(path)

    # extract text based on file type
    if path.suffix.lower() == ".docx":
        text = extract_text_from_docx(path)
    elif path.suffix.lower() == ".pdf":
        text = extract_text_from_pdf(path)
    else:
        return 0

    # skip files with no extractable text
    if not text.strip():
        return 0
    
    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],  # the text chunk
            metadatas=[{"source": str(path), "chunk": i}],  # where it came from
            ids=[f"{path.stem}_{i}"]  # unique ID for this chunk
        )
    
    return len(chunks)

def ingest_directory(data_dir):
    """
    Recursively find and ingest all .docx and .pdf files in a directory.

    Skips files containing keywords associated with low-quality content
    (drafts, downloaded textbooks, system files).
    """
    data_path = Path(data_dir)

    # rglob recursively finds all matching files in all subdirectories
    files = list(data_path.rglob("*.docx")) + list(data_path.rglob("*.pdf"))

    total_chunks = 0
    for file in files:
        # skip draft files, libgen downloads, and mac system files
        if any(skip in str(file).lower() for skip in ["draft", "libgen", ".ds_store"]):
            continue

        print(f"Ingesting: {file.name}")
        chunks = ingest_file(file)
        total_chunks += chunks
        print(f"  -> {chunks} chunks")
    
    print(f"\nDone! Total chunks: {total_chunks}")

if __name__ == "__main__":
    # entry point -> ingest everything in the data/ directory
    ingest_directory("data/")