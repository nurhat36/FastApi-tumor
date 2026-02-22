from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.database import get_db
from app.models.models import Patient,Mask,File
from app.utils.dependencies import get_current_user
import os


router = APIRouter(prefix="/patients", tags=["patients"])


class PatientCreate(BaseModel):
    name: str


@router.post("/")
def create_patient(
    data: PatientCreate,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    new_patient = Patient(
        name=data.name,
        owner_id=current_user.id
    )

    db.add(new_patient)
    db.commit()
    db.refresh(new_patient)

    # Klasör oluştur
    base_path = os.path.join(
        "storage",
        "users",
        str(current_user.id),
        "patients",
        str(new_patient.id)
    )

    os.makedirs(os.path.join(base_path, "originals"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "masks"), exist_ok=True)

    return {
        "id": new_patient.id,
        "name": new_patient.name
    }
@router.get("/")
def get_patients(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    patients = (
        db.query(Patient)
        .filter(Patient.owner_id == current_user.id)
        .all()
    )

    return [
        {
            "id": patient.id,
            "name": patient.name
        }
        for patient in patients
    ]
@router.delete("/{patient_id}")
def delete_patient(
    patient_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    patient = db.query(Patient).filter(
        Patient.id == patient_id,
        Patient.owner_id == current_user.id
    ).first()

    if not patient:
        raise HTTPException(status_code=404, detail="Hasta bulunamadı")

    db.delete(patient)
    db.commit()

    return {"message": "Hasta silindi"}
@router.get("/{patient_id}/images")
def get_patient_images(
    patient_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    # Hasta kontrolü
    patient = db.query(Patient).filter(
        Patient.id == patient_id,
        Patient.owner_id == current_user.id
    ).first()

    if not patient:
        raise HTTPException(status_code=404, detail="Hasta bulunamadı")

    # 🔥 File üzerinden JOIN
    masks = (
        db.query(Mask)
        .join(File, Mask.file_id == File.id)
        .filter(File.patient_id == patient_id)
        .all()
    )

    return [
        {
            "id": mask.id,
            "filename": mask.filename,
            "mask_url": mask.file_path  # Flutter tarafı mask_url bekliyor
        }
        for mask in masks
    ]
