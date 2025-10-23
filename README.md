# 🧠 IKARUS Product Recommendation Backend (ML + Vector DB)

An end-to-end **AI-powered product recommendation system** built with **FastAPI**, integrating **Machine Learning (ML)**, **Computer Vision (CV)**, and **Vector Search (Pinecone)**.  
It provides semantic text-based and visual product recommendations, category classification, and analytics APIs for intelligent e-commerce experiences.

---

## 🚀 Tech Stack

| Component | Technology |
|------------|-------------|
| Backend Framework | FastAPI |
| Text Embedding | SentenceTransformer (`all-MiniLM-L6-v2`) |
| Image Classification | TensorFlow (ResNet-50 fine-tuned) |
| Vector Database | Pinecone |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Deployment | Render |
| Environment | Python 3.11+ |

---

## 📂 Folder Structure

```
Neurostack/
└── backend/
├── app/
│ ├── core/ # Config & Pinecone setup
│ ├── data/ # CSV dataset (intern_data_ikarus.csv)
│ ├── ml/ # ML models (text + image)
│ │ ├── classifier.py # ResNet-50 image model loader
│ │ └── text_model.py # SentenceTransformer text model
│ ├── models/ # Saved model files
│ │ ├── resnet50_best.h5
│ │ └── label_classes.npy
│ ├── routes/ # FastAPI API routes
│ │ ├── text_routes.py
│ │ ├── cv_routes.py
│ │ └── analytics_routes.py
│ ├── scripts/ # Indexing scripts
│ │ └── index_text_data.py
│ └── main.py # App entry point
├── requirements.txt # Dependencies
├── .env # Environment variables
└── README.md # Documentation
```

---

## ⚙️ Environment Setup

### 1️⃣ Clone Repository
```bash
git clone https://github.com/<your-username>/ikarus-backend.git
cd ikarus-backend/Neurostack/backend
```

### 2️⃣ Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate   # (Mac/Linux)
venv\Scripts\activate      # (Windows)
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Create .env file
```bash
# Pinecone configuration
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=us-east-1
TEXT_INDEX=ikarus3d

# Model and data paths
DATA_PATH=./app/data/intern_data_ikarus.csv
RESNET_MODEL_PATH=./app/models/resnet50_best.h5
```
---

## 🧩 Run the Backend Locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

