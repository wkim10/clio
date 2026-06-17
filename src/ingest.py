import os
import chromadb
import docx2txt
import pypdf
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# initialize chromadb
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("clio")

def extract_text_from_docx(path):
    try:
        return docx2txt.process(path)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return ""

def extract_text_from_pdf(path):
    try:
        reader = pypdf.PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return ""

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def ingest_file(path):
    path = Path(path)
    if path.suffix.lower() == ".docx":
        text = extract_text_from_docx(path)
    elif path.suffix.lower() == ".pdf":
        text = extract_text_from_pdf(path)
    else:
        return 0

    if not text.strip():
        return 0
    
    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            metadatas=[{"source": str(path), "chunk": i}],
            ids=[f"{path.stem}_{i}"]
        )
    
    return len(chunks)

def ingest_directory(data_dir):
    data_path = Path(data_dir)
    files = list(data_path.rglob("*.docx")) + list(data_path.rglob("*.pdf"))

    total_chunks = 0
    for file in files:
        if any(skip in str(file).lower() for skip in ["draft", "libgen", ".ds_store"]):
            continue

        print(f"Ingesting: {file.name}")
        chunks = ingest_file(file)
        total_chunks += chunks
        print(f"  -> {chunks} chunks")
    
    print(f"\nDone! Total chunks: {total_chunks}")

if __name__ == "__main__":
    ingest_directory("data/")