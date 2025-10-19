from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import text_routes, cv_routes, analytics_routes

app = FastAPI(title="IKARUS API (Text + CV)", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

app.include_router(text_routes.router)
app.include_router(cv_routes.router)
app.include_router(analytics_routes.router)

@app.get("/")
def root():
    return {"message": "🚀 IKARUS backend running (Text + CV + Analytics)"}
