import faiss 
from sentence_transformers import SentenceTransformer 
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.IndexFlatL2(384)
documents = []


def add_text(text):
    chunks = [text[i:i + 200] for i in range(0, len(text), 200)]
    if not chunks:
        return

    embeddings = model.encode(chunks)
    index.add(embeddings)
    documents.extend(chunks)


def search(query, k=3):
    if len(documents) == 0:
        return []

    q_emb = model.encode([query])
    _, ids = index.search(q_emb, min(k, len(documents)))

    results = []
    for i in ids[0]:
        if i < len(documents):
            results.append(documents[i])

    return results