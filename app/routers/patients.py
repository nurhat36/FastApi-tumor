from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.database import get_db
from app.models.models import Patient
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