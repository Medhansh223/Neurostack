from fastapi import APIRouter
import pandas as pd
from app.core.config import settings

router = APIRouter(prefix="/analytics", tags=["Analytics"])
df = pd.read_csv(settings.DATA_PATH).fillna("")

@router.get("/summary")
def analytics_summary():
    return {
        "total_products": len(df),
        "top_brands": df["brand"].value_counts().head(5).to_dict(),
        "top_categories": df["categories"].value_counts().head(5).to_dict()
    }
