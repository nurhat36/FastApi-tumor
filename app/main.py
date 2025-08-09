from fastapi import FastAPI
from app.routers import auth, segment
from app.database import Base, engine
from fastapi.staticfiles import StaticFiles
import os
app = FastAPI(
    title="Segmentation API",
    description="Tümleşik model ile segmentasyon tahminleri yapar",
    version="1.0.0"
)

# Router'ı dahil et
Base.metadata.create_all(bind=engine)

app.include_router(auth.router)
app.include_router(segment.router)

app.mount("/static", StaticFiles(directory="static"), name="static")
