"""
FastAPI Main Application
Tesla Stock Price Prediction API
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.routers.predict import router as predict_router

# ─── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Tesla Stock Price Prediction API",
    description="ML-powered stock price prediction using Random Forest + XGBoost ensemble",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS (allow all origins for PWA / mobile access) ────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ──────────────────────────────────────────────────────────────────
app.include_router(predict_router)

# ─── Serve Frontend Static Files ─────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        index_path = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"message": "Frontend not found. Place files in /frontend"}

    @app.get("/manifest.json", include_in_schema=False)
    async def serve_manifest():
        return FileResponse(os.path.join(FRONTEND_DIR, "manifest.json"))

    @app.get("/service-worker.js", include_in_schema=False)
    async def serve_sw():
        return FileResponse(os.path.join(FRONTEND_DIR, "service-worker.js"))


@app.get("/ping")
def ping():
    return {"ping": "pong", "service": "Tesla Stock Prediction"}
