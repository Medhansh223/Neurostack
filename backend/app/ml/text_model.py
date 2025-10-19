from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def encode_texts(texts):
    """Return normalized text embeddings."""
    return model.encode(texts, normalize_embeddings=True)
