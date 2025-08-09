# app/routers/segment.py
import io, base64, time
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from starlette.responses import JSONResponse
from PIL import Image
import numpy as np
import cv2
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.unet_model import model
from app.models.models import Mask
from app.utils.security import get_current_user
from app.models.models import User

router = APIRouter(tags=["segment"])

@router.post("/segment")
async def predict_image(file: UploadFile = File(...), db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        contents = await file.read()

        # Görseli aç (grayscale), model inputa hazırla (senin modelin 128x128x1)
        image = Image.open(io.BytesIO(contents)).convert("L")
        original_size = image.size
        image_resized = image.resize((128, 128))
        image_np = np.array(image_resized, dtype=np.float32) / 255.0
        image_np = np.expand_dims(image_np, axis=-1)
        image_np = np.expand_dims(image_np, axis=0)

        # Tahmin
        prediction = model.predict(image_np)[0]
        prediction_mask = (prediction > 0.5).astype(np.uint8) * 255

        # Orijinal boyuta döndür
        prediction_mask_resized = cv2.resize(prediction_mask, original_size)

        # PNG binary'e çevir
        mask_image = Image.fromarray(prediction_mask_resized.astype(np.uint8), mode='L')
        buf = io.BytesIO()
        mask_image.save(buf, format="PNG")
        mask_bytes = buf.getvalue()

        # filename
        filename = f"mask_user{current_user.id}_{int(time.time())}.png"

        # DB'ye kaydet (mask_data olarak binary)
        mask_record = Mask(filename=filename, mask_data=mask_bytes, owner_id=current_user.id)
        db.add(mask_record)
        db.commit()
        db.refresh(mask_record)

        # Base64 döndür (isteğe bağlı)
        mask_base64 = base64.b64encode(mask_bytes).decode('utf-8')

        return JSONResponse(content={
            "mask_id": mask_record.id,
            "filename": filename,
            "mask_base64": mask_base64
        })

    except Exception as e:
        print("HATA segment:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
