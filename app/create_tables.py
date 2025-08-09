# create_tables.py
from app.database import Base, engine
from app.models.models import User, Mask

Base.metadata.drop_all(bind=engine)  # eski tabloları siler
Base.metadata.create_all(bind=engine)  # yeni şema ile oluşturur