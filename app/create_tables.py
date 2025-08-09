# create_tables.py

from app.database import Base, engine
from app.models.models import User, Mask

print("📦 MSSQL tabloları oluşturuluyor...")
Base.metadata.create_all(bind=engine)
print("✅ Tablolar başarıyla oluşturuldu!")
