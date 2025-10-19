import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

class Settings:
    DATA_PATH = os.getenv("DATA_PATH", "./app/data/intern_data_ikarus.csv")
    RESNET_MODEL_PATH = os.getenv("RESNET_MODEL_PATH", "./app/models/resnet50_best.h5")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
    TEXT_INDEX = os.getenv("TEXT_INDEX", "furniture-text")

settings = Settings()

pc = None
if settings.PINECONE_API_KEY:
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    existing = [i["name"] for i in pc.list_indexes()]
    if settings.TEXT_INDEX not in existing:
        pc.create_index(
            name=settings.TEXT_INDEX,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=settings.PINECONE_ENV),
        )
else:
    print("⚠️ Pinecone not configured.")
