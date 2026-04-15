from fastapi import FastAPI
from app.routers import auth, segment, patients, file,mask
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

# DEĞİŞTİRİLEN KISIM: Hepsine prefix="/api" eklendi
app.include_router(auth.router, prefix="/api")
app.include_router(segment.router, prefix="/api")
app.include_router(patients.router, prefix="/api")
app.include_router(file.router, prefix="/api")
app.include_router(mask.router, prefix="/api")

app.mount("/static", StaticFiles(directory="static"), name="static")