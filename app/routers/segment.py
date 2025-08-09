# app/routers/segment.py
import io, time
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from multipart import file_path
from starlette.responses import JSONResponse
from PIL import Image
import numpy as np
import cv2
from sqlalchemy.orm import Session
from pathlib import Path

from app.database import get_db
from app.models.unet_model import model
from app.models.models import Mask
from app.utils.security import get_current_user
from app.models.models import User

router = APIRouter(tags=["segment"])

# STATIC klasör yolu
STATIC_MASKS_DIR = Path("static/masks")
STATIC_MASKS_DIR.mkdir(parents=True, exist_ok=True)  # klasör yoksa oluştur

@router.post("/segment")
async def predict_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        # Kullanıcı dosyayı yükledi -> oku
        contents = await file.read()

        # Görseli grayscale olarak aç
        image = Image.open(io.BytesIO(contents)).convert("L")
        original_size = image.size

        # 128x128'e resize
        image_resized = image.resize((128, 128))
        image_np = np.array(image_resized, dtype=np.float32) / 255.0
        image_np = np.expand_dims(image_np, axis=-1)
        image_np = np.expand_dims(image_np, axis=0)

        # Model tahmini
        prediction = model.predict(image_np)[0]
        prediction_mask = (prediction > 0.5).astype(np.uint8) * 255

        # Orijinal boyuta döndür
        prediction_mask_resized = cv2.resize(prediction_mask, original_size)

        # PNG olarak kaydet
        filename = f"mask_user{current_user.id}_{int(time.time())}.png"
        save_path = STATIC_MASKS_DIR / filename
        mask_image = Image.fromarray(prediction_mask_resized.astype(np.uint8), mode='L')
        mask_image.save(save_path, format="PNG")

        # DB'ye kaydet
        mask_record = Mask(filename=filename, owner_id=current_user.id,file_path=str(save_path))
        db.add(mask_record)
        db.commit()
        db.refresh(mask_record)

        # Mask URL
        mask_url = f"/static/masks/{filename}"

        return JSONResponse(content={
            "mask_id": mask_record.id,
            "filename": filename,
            "mask_url": mask_url
        })

    except Exception as e:
        print("HATA segment:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
