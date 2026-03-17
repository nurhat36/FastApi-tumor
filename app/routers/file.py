from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from pathlib import Path
import time
import shutil
import os

from app.database import get_db
# DİKKAT: Import kısmına Mask eklendi
from app.models.models import File as DBFile, Patient, Mask, User
from app.utils.dependencies import get_current_user

router = APIRouter(tags=["files"])

STATIC_ORIGINALS_DIR = Path("static/originals")
STATIC_ORIGINALS_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/files/{patient_id}")
async def upload_file(
    patient_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # ------------------------------------------------------------
    # 1️⃣ Patient kontrol (kendi hastası mı?)
    # ------------------------------------------------------------
    patient = db.query(Patient).filter(
        Patient.id == patient_id,
        Patient.owner_id == current_user.id
    ).first()

    if not patient:
        raise HTTPException(status_code=404, detail="Hasta bulunamadı.")

    # ------------------------------------------------------------
    # 2️⃣ Dosya kaydet
    # ------------------------------------------------------------
    timestamp = int(time.time())
    filename = f"user{current_user.id}_patient{patient_id}_{timestamp}_{file.filename}"
    file_path = STATIC_ORIGINALS_DIR / filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ------------------------------------------------------------
    # 3️⃣ DB Kaydı
    # ------------------------------------------------------------
    new_file = DBFile(
        filename=filename,
        file_path=str(file_path),
        patient_id=patient_id,
        status="uploaded"
    )

    db.add(new_file)
    db.commit()
    db.refresh(new_file)

    return {
        "id": new_file.id,
        "filename": new_file.filename,
        "status": new_file.status
    }

@router.get("/files/{patient_id}")
def get_files(
    patient_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    patient = db.query(Patient).filter(
        Patient.id == patient_id,
        Patient.owner_id == current_user.id
    ).first()

    if not patient:
        raise HTTPException(status_code=404, detail="Hasta bulunamadı.")

    files = db.query(DBFile).filter(
        DBFile.patient_id == patient_id
    ).order_by(DBFile.uploaded_at.desc()).all()

    result = []
    for f in files:
        # ------------------------------------------------------------
        # YENİ: Bu dosyaya ait üretilmiş bir maske var mı?
        # ------------------------------------------------------------
        mask = db.query(Mask).filter(Mask.file_id == f.id).order_by(Mask.created_at.desc()).first()

        result.append({
            "id": f.id,
            "filename": f.filename,
            # Eğer maske varsa status'ü zorla 'segmented' yap, yoksa orijinalini ver
            "status": "segmented" if mask else f.status,
            "file_path": f.file_path,
            # Eğer maske varsa URL'sini ver ki React/Flutter otomatik yüklesin
            "mask_url": f"/{mask.file_path}" if mask else None
        })

    return result
@router.delete("/files/{file_id}")
async def delete_full_file(
    file_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # 1. Dosyayı veritabanında bul
    db_file = db.query(DBFile).filter(DBFile.id == file_id).first()
    if not db_file:
        raise HTTPException(status_code=404, detail="Dosya bulunamadı.")

    # 2. Bu dosyaya bağlı maskeleri bul ve diskten sil
    masks = db.query(Mask).filter(Mask.file_id == file_id).all()
    for mask in masks:
        if os.path.exists(mask.file_path):
            os.remove(mask.file_path)
        db.delete(mask) # Veritabanından da sil

    # 3. Ana MR dosyasını diskten sil
    if os.path.exists(db_file.file_path):
        os.remove(db_file.file_path)

    # 4. Ana dosyayı veritabanından sil ve işlemi onayla
    db.delete(db_file)
    db.commit()

    return {"message": "Dosya ve bağlı tüm maskeler kalıcı olarak silindi."}