import os
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load environment
load_dotenv()
DATA_PATH = os.getenv("DATA_PATH", "./app/data/intern_data_ikarus.csv")
INDEX_NAME = os.getenv("TEXT_INDEX", "ikarus3d")
API_KEY = os.getenv("PINECONE_API_KEY")

# 1️⃣ Initialize Pinecone and Model
pc = Pinecone(api_key=API_KEY)
index = pc.Index(INDEX_NAME)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 2️⃣ Load your dataset
df = pd.read_csv(DATA_PATH).fillna("")
print(f"✅ Loaded {len(df)} products from dataset")

# 3️⃣ Combine relevant text fields
text_fields = ["title", "brand", "description", "categories", "material", "color"]
df["combined_text"] = df[text_fields].astype(str).agg(" ".join, axis=1)

# 4️⃣ Generate embeddings and upload to Pinecone
batch_size = 100
for i in tqdm(range(0, len(df), batch_size), desc="Indexing to Pinecone"):
    batch = df.iloc[i : i + batch_size]
    vectors = []
    embeddings = model.encode(batch["combined_text"].tolist(), normalize_embeddings=True)
    for j, row in enumerate(batch.itertuples()):
        meta = {col: getattr(row, col) for col in df.columns}
        vectors.append({
            "id": str(row.uniq_id) if hasattr(row, "uniq_id") else f"prod_{i+j}",
            "values": embeddings[j].tolist(),
            "metadata": meta
        })
    index.upsert(vectors=vectors)

print(f"✅ Successfully indexed {len(df)} products into Pinecone index '{INDEX_NAME}'")