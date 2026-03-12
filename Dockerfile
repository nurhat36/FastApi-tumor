# Python 3.10 tabanlı hafif imaj
FROM python:3.10-slim

# Çalışma dizini
WORKDIR /app

# İşletim sistemi seviyesindeki bağımlılıklar (Güncel paket isimleri)
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 libpq-dev gcc && rm -rf /var/lib/apt/lists/*

# Kütüphaneleri kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tüm kodları kopyala
COPY . .

# Portu aç
EXPOSE 8000

# Uygulamayı başlat
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]