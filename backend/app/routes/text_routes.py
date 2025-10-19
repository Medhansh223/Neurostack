from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.ml.text_model import encode_texts
from app.core.config import pc, settings

router = APIRouter(prefix="/recommend", tags=["Text Recommendation"])

class TextQuery(BaseModel):
    query: str
    top_k: int = 5

@router.post("/text")
def recommend_text(query: TextQuery):
    if not pc:
        raise HTTPException(400, "Pinecone not configured.")
    index = pc.Index(settings.TEXT_INDEX)
    q_vec = encode_texts([query.query])[0].tolist()
    res = index.query(vector=q_vec, top_k=query.top_k, include_metadata=True)

    recommendations = []
    for match in res.get("matches", []):
        meta = match.get("metadata", {})
        recommendations.append({
            "title": meta.get("title"),
            "brand": meta.get("brand"),
            "price": meta.get("price"),
            "image": meta.get("images", "").split(",")[0],
            "score": round(float(match.get("score", 0)), 3)
        })

    return {"query": query.query, "recommendations": recommendations}
