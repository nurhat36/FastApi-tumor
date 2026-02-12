# app/routers/segment.py
import io, time
import os
import shutil

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
import uuid
import nibabel as nib
from fastapi import Body




from fastapi import APIRouter, File, UploadFile, Form, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from PIL import Image, ImageDraw
import numpy as np
import io, cv2, time, os
from pathlib import Path

# ============================================================
# 📦 Router ve dizin ayarları
# ============================================================
router = APIRouter(tags=["segment"])

STATIC_MASKS_DIR = Path("static/masks")
STATIC_ORIGINALS_DIR = Path("static/originals")
STATIC_MASKS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_ORIGINALS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 🧮 Dice Loss (Model ile aynı olmalı)
# ============================================================
def dice_loss(y_true, y_pred):
    smooth = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1.0 - dice

# ============================================================
# 🧠 Model Yükleme
# ============================================================
MODEL_PATH = "app/static/best_model.h5"

print("🧠 Model yükleniyor...")
model = load_model(MODEL_PATH, custom_objects={"dice_loss": dice_loss},compile=False)
print("✅ Model başarıyla yüklendi.")

# ============================================================
# 🎯 Segmentasyon Endpoint
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
        # ------------------------------------------------------------
        # 🔹 1. Dosyayı oku ve griye çevir
        # ------------------------------------------------------------
        contents = await file.read()

        # 🔥 Eğer NIfTI ise
        if file.filename.endswith((".nii", ".nii.gz")):
            mask_filename = segment_volume_nifti(
                contents,
                file.filename,
                current_user
            )

            return {
                "type": "volume",
                "mask_filename": mask_filename,
                "mask_url": f"/static/masks/{mask_filename}"
            }

        # 🔥 Değilse PNG gibi devam et
        image = Image.open(io.BytesIO(contents)).convert("L")

        original_size = image.size  # (width, height)

        # Orijinal görseli kaydet
        original_filename = f"original_user{current_user.id}_{int(time.time())}.png"
        original_save_path = STATIC_ORIGINALS_DIR / original_filename
        image.save(original_save_path, format="PNG")

        # ------------------------------------------------------------
        # 🔹 2. Görseli modele uygun hale getir
        # ------------------------------------------------------------


        # ✅ 2 Kanallı giriş oluştur (örneğin FLAIR + T1CE yoksa aynı kanal iki kez kopyalanır)
        # ------------------------------------------------------------
        # 🔹 2. Görseli modele uygun hale getir
        # ------------------------------------------------------------
        IMG_SIZE = 128
        image_resized = image.resize((IMG_SIZE, IMG_SIZE))
        image_np = np.array(image_resized, dtype=np.float32) / 255.0

        # ❌ ESKİ HATALI KISIM: (image_np, image_np) yapıp 2 kanal yapıyordun.
        # image_np = np.stack((image_np, image_np), axis=-1)

        # ✅ YENİ DOĞRU KISIM: Tek kanal (128, 128, 1) haline getiriyoruz.
        image_np = np.expand_dims(image_np, axis=-1)  # Şekil: (128, 128, 1) olur
        image_np = np.expand_dims(image_np, axis=0)  # Şekil: (1, 128, 128, 1) olur (Batch boyutu eklendi)


        # ------------------------------------------------------------
        # 🔹 3. Model Tahmini
        # ------------------------------------------------------------
        prediction = model.predict(image_np)[0]  # (128,128,1) veya (128,128,2)

        # Eğer çok kanallı maske varsa, ilk kanalı al
        if prediction.ndim == 3 and prediction.shape[-1] > 1:
            prediction = prediction[..., 0]

        # Eşikleme
        prediction_mask = (prediction > 0.5).astype(np.uint8) * 255

        # Eğer squeeze yapılabilecek eksen varsa
        if prediction_mask.ndim == 3 and prediction_mask.shape[-1] == 1:
            prediction_mask = np.squeeze(prediction_mask, axis=-1)

        # ------------------------------------------------------------
        # 🔹 4. Maskeyi orijinal boyuta geri döndür
        # ------------------------------------------------------------
        prediction_mask_resized = cv2.resize(prediction_mask, original_size)

        # ------------------------------------------------------------
        # 🔹 5. Kullanıcının çizdiği bölge varsa uygula
        # ------------------------------------------------------------
        if width > 0 and height > 0:
            mask = Image.new("L", original_size, 0)
            draw = ImageDraw.Draw(mask)
            if shape == "rectangle":
                draw.rectangle([x, y, x + width, y + height], fill=255)
            elif shape in ["circle", "oval"]:
                draw.ellipse([x, y, x + width, y + height], fill=255)
            mask_np = np.array(mask)
            prediction_mask_resized = cv2.bitwise_and(prediction_mask_resized, mask_np)

        # ------------------------------------------------------------
        # 🔹 6. Maskeyi kaydet
        # ------------------------------------------------------------
        mask_filename = f"mask_user{current_user.id}_{int(time.time())}.png"
        mask_save_path = STATIC_MASKS_DIR / mask_filename

        mask_image = Image.fromarray(prediction_mask_resized.astype(np.uint8), mode='L')
        mask_image.save(mask_save_path, format="PNG")

        # ------------------------------------------------------------
        # 🔹 7. Veritabanına kaydet
        # ------------------------------------------------------------
        mask_record = Mask(
            filename=mask_filename,
            file_path=str(mask_save_path),
            original_file_path=str(original_save_path),
            owner_id=current_user.id
        )
        db.add(mask_record)
        db.commit()
        db.refresh(mask_record)

        # ------------------------------------------------------------
        # 🔹 8. JSON yanıt döndür
        # ------------------------------------------------------------
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
def segment_volume_nifti(file_bytes, original_filename, current_user):

    ext = ".nii.gz" if original_filename.endswith(".nii.gz") else ".nii"

    temp_filename = f"temp_{uuid.uuid4().hex}{ext}"
    temp_path = STATIC_MASKS_DIR / temp_filename

    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    nii = nib.load(str(temp_path))
    volume = nii.get_fdata()
    affine = nii.affine

    mask_slices = []

    for i in range(volume.shape[2]):
        slice_img = volume[:, :, i]

        slice_img = np.clip(
            slice_img,
            np.percentile(slice_img, 1),
            np.percentile(slice_img, 99)
        )

        slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8)
        slice_img = (slice_img * 255).astype(np.uint8)

        pil_slice = Image.fromarray(slice_img).convert("L")
        mask_slice = segment_slice_with_model(pil_slice)

        mask_slices.append(mask_slice)

    mask_volume = np.stack(mask_slices, axis=2)

    timestamp = int(time.time())
    mask_filename = f"mask_volume_user{current_user.id}_{timestamp}.nii.gz"
    mask_save_path = STATIC_MASKS_DIR / mask_filename

    mask_nifti = nib.Nifti1Image(mask_volume, affine)
    nib.save(mask_nifti, str(mask_save_path))

    os.remove(temp_path)

    return mask_filename



def segment_slice_with_model(pil_image):
    IMG_SIZE = 128

    original_size = pil_image.size

    image_resized = pil_image.resize((IMG_SIZE, IMG_SIZE))
    image_np = np.array(image_resized, dtype=np.float32) / 255.0

    image_np = np.expand_dims(image_np, axis=-1)
    image_np = np.expand_dims(image_np, axis=0)

    prediction = model.predict(image_np)[0]

    if prediction.ndim == 3 and prediction.shape[-1] > 1:
        prediction = prediction[..., 0]

    prediction_mask = (prediction > 0.5).astype(np.uint8) * 255

    if prediction_mask.ndim == 3 and prediction_mask.shape[-1] == 1:
        prediction_mask = np.squeeze(prediction_mask, axis=-1)

    prediction_mask_resized = cv2.resize(prediction_mask, original_size)

    return prediction_mask_resized




@router.post("/segment/manual")
async def create_manual_mask(
    original_file: UploadFile = File(...),
    mask_file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        timestamp = int(time.time())

        # -------------------------------------------------
        # 📁 Dosya isimleri
        # -------------------------------------------------
        original_filename = f"original_user{current_user.id}_{timestamp}.png"
        mask_filename = f"manual_mask_user{current_user.id}_{timestamp}.png"

        original_save_path = STATIC_ORIGINALS_DIR / original_filename
        mask_save_path = STATIC_MASKS_DIR / mask_filename

        # -------------------------------------------------
        # 🖼 Orijinal resmi kaydet
        # -------------------------------------------------
        original_contents = await original_file.read()
        with open(original_save_path, "wb") as f:
            f.write(original_contents)

        # -------------------------------------------------
        # 🎨 Maskeyi kaydet (grayscale garanti)
        # -------------------------------------------------
        mask_contents = await mask_file.read()
        mask_image = Image.open(io.BytesIO(mask_contents)).convert("L")
        mask_image.save(mask_save_path, format="PNG")

        # -------------------------------------------------
        # 🗄 DB kaydı oluştur
        # -------------------------------------------------
        mask_record = Mask(
            filename=mask_filename,
            file_path=str(mask_save_path),
            original_file_path=str(original_save_path),
            owner_id=current_user.id
        )

        db.add(mask_record)
        db.commit()
        db.refresh(mask_record)

        return {
            "mask_id": mask_record.id,
            "mask_url": f"/static/masks/{mask_filename}",
            "original_url": f"/static/originals/{original_filename}",
            "type": "manual"
        }

    except Exception as e:
        print("❌ Manual segment error:", str(e))
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
async def update_mask(  # I/O işlemi olduğu için async yapıyoruz
        mask_id: int,
        file: UploadFile = File(...),  # Body yerine File alıyoruz
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    # 1. Kaydı bul
    mask_record = db.query(Mask).filter(
        Mask.id == mask_id,
        Mask.owner_id == current_user.id
    ).first()

    if not mask_record:
        raise HTTPException(status_code=404, detail="Mask bulunamadı veya size ait değil.")

    # 2. Dosya kaydetme işlemleri
    # Eski dosya varsa silebilirsin veya üzerine yazabilirsin.
    # Burada güvenli bir dosya adı oluşturuyoruz (çakışmayı önlemek için UUID veya mask_id kullanabilirsin)
    file_extension = ".png"  # Flutter'dan PNG göndereceğiz
    new_filename = f"mask_{mask_id}_{uuid.uuid4().hex[:8]}{file_extension}"

    save_directory = "static/masks/"
    os.makedirs(save_directory, exist_ok=True)  # Klasör yoksa oluştur

    file_location = os.path.join(save_directory, new_filename)

    # Dosyayı diske yaz
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)

    # 3. Veritabanını güncelle
    # Eğer eski dosya adı farklıysa eskisini diskten silme kodu buraya eklenebilir.
    mask_record.filename = new_filename
    mask_record.file_path=f"/static/masks/{mask_record.filename}"
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
