from fastapi import FastAPI
from app.routers import segment

app = FastAPI()

app.include_router(segment.router)
