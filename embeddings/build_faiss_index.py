import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import os

DATA_PATH = "data/shl_catalog.csv"
MODEL_NAME = "all-MiniLM-L6-v2"

def main():
    df = pd.read_csv(DATA_PATH)

    # Combine text fields for embedding
    df["combined_text"] = (
        df["name"].fillna("") + " " +
        df["description"].fillna("") + " " +
        df["test_type"].fillna("")
    )

    texts = df["combined_text"].tolist()

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    print("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs("embeddings/store", exist_ok=True)

    faiss.write_index(index, "embeddings/store/shl_faiss.index")

    with open("embeddings/store/metadata.pkl", "wb") as f:
        pickle.dump(df.to_dict(orient="records"), f)

    print("FAISS index & metadata saved successfully")

if __name__ == "__main__":
    main()
