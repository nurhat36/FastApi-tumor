# app/routers/segment.py
import io, time
import os

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from multipart import file_path
from sqlalchemy import Double
from starlette.responses import JSONResponse
from PIL import Image, ImageDraw
import numpy as np
import cv2
from sqlalchemy.orm import Session
from pathlib import Path

from app.database import get_db
from app.models.unet_model import model
from app.models.models import Mask
from app.utils.security import get_current_user
from app.models.models import User
from fastapi import Body
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K


router = APIRouter(tags=["segment"])

# STATIC klasör yolu
STATIC_MASKS_DIR = Path("static/masks")
STATIC_ORIGINALS_DIR = Path("static/originals")
STATIC_MASKS_DIR.mkdir(parents=True, exist_ok=True)  # klasör yoksa oluştur
STATIC_ORIGINALS_DIR.mkdir(parents=True, exist_ok=True)


from fastapi import APIRouter, File, UploadFile, Form, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from PIL import Image, ImageDraw
import numpy as np
import io, cv2, time

def dice_loss(y_true, y_pred):
    smooth = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1.0 - dice

# 🔹 Eğitimde kullandığın model dosyası yolu
MODEL_PATH = "app/static/tumor_segmentation_model.h5"

print("🧠 Model yükleniyor...")
model = load_model(MODEL_PATH, custom_objects={"dice_loss": dice_loss})
print("✅ Model başarıyla yüklendi.")

# 🔹 Kaydedilecek klasör yolları
STATIC_ORIGINALS_DIR = "static/originals"
STATIC_MASKS_DIR = "static/masks"

# ============================================================
# 🧩 Segmentasyon Endpoint
# ============================================================
@router.post("/segment")
async def predict_image(
    file: UploadFile = File(...),
    x: float = Form(...),
    y: float = Form(...),
    width: float = Form(...),
    height: float = Form(...),
    shape: str = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        # 🔹 Dosyayı oku ve Pillow ile griye çevir
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("L")
        original_size = image.size

        # 🔹 Orijinal görseli kaydet
        original_filename = f"original_user{current_user.id}_{int(time.time())}.png"
        original_save_path = f"{STATIC_ORIGINALS_DIR}/{original_filename}"
        image.save(original_save_path, format="PNG")

        # ====================================================
        # 🔹 Model için görseli hazırla
        # ====================================================
        IMG_SIZE = 128
        image_resized = image.resize((IMG_SIZE, IMG_SIZE))
        image_np = np.array(image_resized, dtype=np.float32) / 255.0
        image_np = np.expand_dims(image_np, axis=(0, -1))  # (1,128,128,1)

        # 🔹 Model tahmini
        prediction = model.predict(image_np)[0]
        prediction_mask = (prediction > 0.5).astype(np.uint8) * 255

        # 🔹 Maskeyi orijinal boyuta döndür
        prediction_mask_resized = cv2.resize(prediction_mask, original_size)

        # ====================================================
        # 🔹 Eğer kullanıcı bir bölge seçtiyse (rectangle / circle)
        # ====================================================
        if width > 0 and height > 0:
            mask = Image.new("L", original_size, 0)
            draw = ImageDraw.Draw(mask)
            if shape == "rectangle":
                draw.rectangle([x, y, x + width, y + height], fill=255)
            elif shape in ["circle", "oval"]:
                draw.ellipse([x, y, x + width, y + height], fill=255)
            mask_np = np.array(mask)
            prediction_mask_resized = cv2.bitwise_and(prediction_mask_resized, mask_np)

        # ====================================================
        # 🔹 Maskeyi kaydet
        # ====================================================
        mask_filename = f"mask_user{current_user.id}_{int(time.time())}.png"
        mask_save_path = f"{STATIC_MASKS_DIR}/{mask_filename}"

        mask_image = Image.fromarray(prediction_mask_resized.astype(np.uint8), mode='L')
        mask_image.save(mask_save_path, format="PNG")

        # ====================================================
        # 🔹 Veritabanı kaydı
        # ====================================================
        mask_record = Mask(
            filename=mask_filename,
            file_path=str(mask_save_path),
            original_file_path=str(original_save_path),
            owner_id=current_user.id
        )
        db.add(mask_record)
        db.commit()
        db.refresh(mask_record)

        # ====================================================
        # 🔹 Yanıt
        # ====================================================
        return JSONResponse(content={
            "mask_id": mask_record.id,
            "mask_filename": mask_filename,
            "mask_url": f"/static/masks/{mask_filename}",
            "original_filename": original_filename,
            "original_url": f"/static/originals/{original_filename}"
        })

    except Exception as e:
        print("❌ HATA segment:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/my-masks")
def get_my_segmented_images(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        masks = db.query(Mask).filter(Mask.owner_id == current_user.id).all()

        results = []
        for mask in masks:
            results.append({
                "mask_id": mask.id,
                "filename": mask.filename,
                "mask_url": f"/static/masks/{mask.filename}"
            })

        return JSONResponse(content=results)

    except Exception as e:
        print("HATA get_my_segmented_images:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/segment/{mask_id}")
def delete_mask(
    mask_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Mask kaydını bul
    mask_record = db.query(Mask).filter(
        Mask.id == mask_id,
        Mask.owner_id == current_user.id
    ).first()

    if not mask_record:
        raise HTTPException(status_code=404, detail="Mask bulunamadı veya size ait değil.")

    # Dosyaları sil
    try:
        if mask_record.file_path and os.path.exists(mask_record.file_path):
            os.remove(mask_record.file_path)

        if mask_record.original_file_path and os.path.exists(mask_record.original_file_path):
            os.remove(mask_record.original_file_path)
    except Exception as e:
        print("Dosya silme hatası:", str(e))



    # DB kaydını sil
    db.delete(mask_record)
    db.commit()

    return {"detail": f"Mask (id={mask_id}) ve ilgili dosyalar silindi."}
@router.put("/segment/{mask_id}")
def update_mask(
    mask_id: int,
    filename: str = Body(..., embed=True),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    mask_record = db.query(Mask).filter(
        Mask.id == mask_id,
        Mask.owner_id == current_user.id
    ).first()

    if not mask_record:
        raise HTTPException(status_code=404, detail="Mask bulunamadı veya size ait değil.")

    mask_record.filename = filename
    db.commit()
    db.refresh(mask_record)

    return {
        "mask_id": mask_record.id,
        "filename": mask_record.filename,
        "mask_url": f"/static/masks/{mask_record.filename}"
    }
@router.put("/segment/{mask_id}/replace")
async def replace_mask(
    mask_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    mask_record = db.query(Mask).filter(
        Mask.id == mask_id,
        Mask.owner_id == current_user.id
    ).first()

    if not mask_record:
        raise HTTPException(status_code=404, detail="Mask bulunamadı veya size ait değil.")

    # Eski dosyaları sil
    try:
        if mask_record.file_path and os.path.exists(mask_record.file_path):
            os.remove(mask_record.file_path)
    except Exception as e:
        print("Eski mask dosyası silinemedi:", str(e))

    # Yeni dosyayı kaydet
    contents = await file.read()
    new_filename = f"mask_user{current_user.id}_{int(time.time())}.png"
    new_save_path = STATIC_MASKS_DIR / new_filename

    with open(new_save_path, "wb") as f:
        f.write(contents)

    # DB kaydını güncelle
    mask_record.filename = new_filename
    mask_record.file_path = str(new_save_path)

    db.commit()
    db.refresh(mask_record)

    return {
        "mask_id": mask_record.id,
        "filename": mask_record.filename,
        "mask_url": f"/static/masks/{mask_record.filename}"
    }
