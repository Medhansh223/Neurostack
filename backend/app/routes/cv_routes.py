from fastapi import APIRouter, UploadFile, File, HTTPException
from app.ml.classifier import predict_category

router = APIRouter(prefix="/predict", tags=["Computer Vision"])

@router.post("/category")
async def predict_category_route(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = predict_category(image_bytes)
    if result is None:
        raise HTTPException(400, "ResNet model not available or invalid.")
    return {"predicted_category_index": result}
