from fastapi import FastAPI
from app.routers import auth, segment, patients, file
from app.database import Base, engine
from fastapi.staticfiles import StaticFiles
# 1. Bunu ekle
from fastapi.middleware.cors import CORSMiddleware

import os

app = FastAPI(
    title="Segmentation API",
    description="Tümleşik model ile segmentasyon tahminleri yapar",
    version="1.0.0"
)

# 2. CORS AYARLARINI BURAYA EKLE (Router'lardan önce olması kritiktir)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Geliştirme aşamasında her yerden gelen isteğe izin verir
    allow_credentials=True,
    allow_methods=["*"],  # GET, POST, PUT, DELETE vb. hepsine izin verir
    allow_headers=["*"],  # Authentication (Bearer token) header'larına izin verir
)

# Router'ı dahil et
Base.metadata.create_all(bind=engine)

app.include_router(auth.router)
app.include_router(segment.router)
app.include_router(patients.router)
app.include_router(file.router)

app.mount("/static", StaticFiles(directory="static"), name="static")