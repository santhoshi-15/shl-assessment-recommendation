import sys
import os

# make project root visible
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI
from pydantic import BaseModel
from retrieval.recommend import recommend

app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="RAG-based assessment recommendation system",
    version="1.0"
)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 10


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend")
def recommend_assessments(req: QueryRequest):
    results = recommend(req.query, top_k=req.top_k)

    return {
        "query": req.query,
        "top_k": req.top_k,
        "recommendations": [
            {
                "name": r["name"],
                "url": r["url"],
                "test_type": r["test_type"]
            }
            for r in results
        ]
    }
