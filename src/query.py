import chromadb
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("clio")
anthropic_client = anthropic.Anthropic()

def query(question, n_results=5):
    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )

    context = "\n\n".join(results["documents"][0])
    sources = [r["source"] for r in results["metadatas"][0]]

    prompt = f"""You are Clio, a personal history research assistant. 
Answer the question based on the context provided from the user's personal history notes and documents.

Context:
{context}

Question: {question}

Answer thoughtfully and cite which sources you drew from."""
    
    response = anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    print("\n--- Answer ---")
    print(response.content[0].text)
    print("\n--- Sources ---")
    for source in set(sources):
        print(f"  {source}")

if __name__ == "__main__":
    while True:
        question = input("\nAsk Clio: ")
        if question.lower() in ["quit", "exit"]:
            break
        query(question)