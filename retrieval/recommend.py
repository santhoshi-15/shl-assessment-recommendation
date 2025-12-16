import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

def load_resources():
    index = faiss.read_index("embeddings/store/shl_faiss.index")
    with open("embeddings/store/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    model = SentenceTransformer(MODEL_NAME)
    return index, metadata, model

def is_behavioral_query(query):
    keywords = ["communication", "team", "collaboration", "leadership", "behavior"]
    return any(k in query.lower() for k in keywords)

def recommend(query, top_k=10):
    index, metadata, model = load_resources()

    query_vec = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, 20)

    candidates = [metadata[i] for i in indices[0]]

    technical = []
    behavioral = []

    for item in candidates:
        if "Personality" in item["test_type"]:
            behavioral.append(item)
        else:
            technical.append(item)

    results = []

    if is_behavioral_query(query):
        results.extend(technical[:5])
        results.extend(behavioral[:5])
    else:
        results.extend(technical[:top_k])

    return results[:top_k]

if __name__ == "__main__":
    query = "Looking for a Java developer with good communication skills"
    recs = recommend(query)

    for r in recs:
        print(r["name"], "->", r["url"])
