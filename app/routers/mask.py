from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.models import File as DBFile, Patient, Mask, User
from app.utils.dependencies import get_current_user

router = APIRouter(
    prefix="/masks",
    tags=["masks"]
)


@router.get("/file/{file_id}")
async def get_masks_by_file(
        file_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    # ------------------------------------------------------------
    # 1️⃣ Ana Dosyayı Bul
    # ------------------------------------------------------------
    file_record = db.query(DBFile).filter(DBFile.id == file_id).first()

    if not file_record:
        raise HTTPException(status_code=404, detail="Ana MR dosyası bulunamadı.")

    # ------------------------------------------------------------
    # 2️⃣ Güvenlik: Dosyanın ait olduğu hasta bu doktora mı ait?
    # ------------------------------------------------------------
    patient = db.query(Patient).filter(
        Patient.id == file_record.patient_id,
        Patient.owner_id == current_user.id
    ).first()

    if not patient:
        raise HTTPException(status_code=403, detail="Bu dosyanın maskelerine erişim yetkiniz yok.")

    # ------------------------------------------------------------
    # 3️⃣ Maskeleri Çek (En yeni en üstte)
    # ------------------------------------------------------------
    masks = db.query(Mask).filter(Mask.file_id == file_id).order_by(Mask.created_at.desc()).all()

    return [
        {
            "id": m.id,
            "filename": m.filename,
            "mask_url": f"/{m.file_path}",  # Frontend'in doğrudan kullanabilmesi için başına / koyuyoruz
            "created_at": m.created_at
        }
        for m in masks
    ]