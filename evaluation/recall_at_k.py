import sys
import os

# make project root visible
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from retrieval.recommend import recommend

EXCEL_PATH = "Gen_AI Dataset.xlsx"
TRAIN_SHEET = "Train-Set"
TOP_K = 10


def recall_at_k(predicted, relevant, k=10):
    if len(relevant) == 0:
        return 0.0
    hits = len(set(predicted[:k]) & set(relevant))
    return hits / len(relevant)


def normalize_url(url):
    if not isinstance(url, str):
        return ""
    url = url.strip().lower()
    url = url.replace("https://www.shl.com", "")
    url = url.replace("http://www.shl.com", "")
    if url.endswith("/"):
        url = url[:-1]
    return url


def main():
    df = pd.read_excel(EXCEL_PATH, sheet_name=TRAIN_SHEET)

    print("📌 Columns found in Train-Set:")
    print(df.columns.tolist())

    query_col = "Query"
    relevant_col = "Assessment_url"

    print("✅ Using query column:", query_col)
    print("✅ Using relevant column:", relevant_col)
    print("-" * 40)

    recalls = []

    grouped = df.groupby(query_col)

    for i, (query, group) in enumerate(grouped, start=1):
        relevant_urls = [
            normalize_url(u)
            for u in group[relevant_col].dropna().tolist()
        ]

        recs = recommend(query, top_k=TOP_K)
        predicted_urls = [
            normalize_url(r["url"]) for r in recs
        ]

        r10 = recall_at_k(predicted_urls, relevant_urls, TOP_K)
        recalls.append(r10)

        print(f"Query {i}: Recall@10 = {r10:.2f}")

    mean_recall = sum(recalls) / len(recalls)
    print("\n==============================")
    print(f"MEAN RECALL@10 = {mean_recall:.4f}")
    print("==============================")


if __name__ == "__main__":
    main()
