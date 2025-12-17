import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from retrieval.recommend import recommend

EXCEL_PATH = "Gen_AI Dataset.xlsx"
TEST_SHEET = "Test-Set"
TOP_K = 10


def main():
    df = pd.read_excel(EXCEL_PATH, sheet_name=TEST_SHEET)

    print("Columns in Test-Set:", df.columns.tolist())

    query_col = [c for c in df.columns if "query" in c.lower()][0]

    rows = []

    for _, row in df.iterrows():
        query = row[query_col]
        recs = recommend(query, top_k=TOP_K)

        for r in recs:
            rows.append({
                "Query": query,
                "Assessment_url": r["url"]
            })

    os.makedirs("submission", exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv("submission/test_predictions.csv", index=False)

    print("✅ submission/test_predictions.csv generated")
    print("Total rows:", len(out_df))


if __name__ == "__main__":
    main()
