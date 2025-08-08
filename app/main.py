from fastapi import FastAPI
from app.routers import segment

app = FastAPI(
    title="Segmentation API",
    description="Tümleşik model ile segmentasyon tahminleri yapar",
    version="1.0.0"
)

# Router'ı dahil et
app.include_router(segment.router)
