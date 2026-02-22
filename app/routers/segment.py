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
from app.models.models import User,File as DBFile,Patient,Mask
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
from starlette.responses import StreamingResponse

# ============================================================
# 📦 Router ve dizin ayarları
# ============================================================
# ============================================================
# 📦 Router ve Dizin Ayarları (Production Ready)
# ============================================================

from fastapi import APIRouter
from pathlib import Path
import os

router = APIRouter(tags=["segment"])

# ------------------------------------------------------------
# 🔹 Base Path (Güvenli Path Çözümü)
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

STATIC_MASKS_DIR = Path("static/masks")
STATIC_MASKS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 🧮 Dice Loss (Model ile aynı olmalı)
# ============================================================



def dice_loss(y_true, y_pred):
    smooth = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + smooth
    )
    return 1.0 - dice


# ============================================================
# 🧠 Model Yükleme (Startup Event ile)
# ============================================================



MODEL_PATH = BASE_DIR / "static" / "best_model.h5"

model = None  # Global model referansı


@router.on_event("startup")
def load_segmentation_model():
    global model

    if model is None:
        print("🧠 Segmentasyon modeli yükleniyor...")

        if not MODEL_PATH.exists():
            raise RuntimeError(f"Model bulunamadı: {MODEL_PATH}")

        model = load_model(
            str(MODEL_PATH),
            custom_objects={"dice_loss": dice_loss},
            compile=False
        )

        print("✅ Segmentasyon modeli başarıyla yüklendi.")


# ============================================================
# 🔎 Model Getter (Güvenli erişim için)
# ============================================================

def get_model():
    if model is None:
        raise RuntimeError("Model henüz yüklenmedi.")
    return model

# ============================================================
# 🎯 Segmentasyon Endpoint
# ============================================================
# ============================================================
# 🎯 Segmentasyon Endpoint (YENİ MİMARİ)
# ============================================================
@router.post("/segment/{file_id}")
async def predict_image(
    file_id: int,
    x: float = Form(0),
    y: float = Form(0),
    width: float = Form(0),
    height: float = Form(0),
    shape: str = Form("rectangle"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        # ------------------------------------------------------------
        # 🔹 1. File kaydını bul + yetki kontrolü
        # ------------------------------------------------------------
        file_record = (
            db.query(DBFile)
            .join(DBFile.patient)
            .filter(
                DBFile.id == file_id,
                Patient.owner_id == current_user.id
            )
            .first()
        )

        if not file_record:
            raise HTTPException(status_code=404, detail="Dosya bulunamadı.")

        if not os.path.exists(file_record.file_path):
            raise HTTPException(status_code=404, detail="Orijinal dosya diskte yok.")

        # ------------------------------------------------------------
        # 🔹 2. Dosyayı oku
        # ------------------------------------------------------------
        with open(file_record.file_path, "rb") as f:
            contents = f.read()

        # ============================================================
        # 🔥 NIFTI VOLUME SEGMENTATION
        # ============================================================
        if file_record.file_path.endswith((".nii", ".nii.gz")):

            mask_filename = segment_volume_nifti(
                contents,
                file_record.filename,
                current_user
            )

            mask_save_path = STATIC_MASKS_DIR / mask_filename

            # --------------------------------------------------------
            # 🗄 Mask DB kaydı (file_id ile bağlı!)
            # --------------------------------------------------------
            mask_record = Mask(
                filename=mask_filename,
                file_path=str(mask_save_path),
                file_id=file_record.id
            )

            file_record.status = "segmented"

            db.add(mask_record)
            db.commit()
            db.refresh(mask_record)

            return {
                "type": "volume",
                "mask_id": mask_record.id,
                "mask_url": f"/static/masks/{mask_filename}"
            }

        # ============================================================
        # 🔥 2D IMAGE SEGMENTATION
        # ============================================================
        image = Image.open(io.BytesIO(contents)).convert("L")
        original_size = image.size

        # ------------------------------------------------------------
        # 🔹 3. Model input hazırlığı
        # ------------------------------------------------------------
        IMG_SIZE = 128
        image_resized = image.resize((IMG_SIZE, IMG_SIZE))
        image_np = np.array(image_resized, dtype=np.float32) / 255.0
        image_np = np.expand_dims(image_np, axis=-1)
        image_np = np.expand_dims(image_np, axis=0)

        # ------------------------------------------------------------
        # 🔹 4. Model Tahmini
        # ------------------------------------------------------------
        prediction = model.predict(image_np)[0]

        if prediction.ndim == 3 and prediction.shape[-1] > 1:
            prediction = prediction[..., 0]

        prediction_mask = (prediction > 0.5).astype(np.uint8) * 255

        if prediction_mask.ndim == 3 and prediction_mask.shape[-1] == 1:
            prediction_mask = np.squeeze(prediction_mask, axis=-1)

        prediction_mask_resized = cv2.resize(prediction_mask, original_size)

        # ------------------------------------------------------------
        # 🔹 5. Kullanıcı ROI uygulaması
        # ------------------------------------------------------------
        if width > 0 and height > 0:
            mask = Image.new("L", original_size, 0)
            draw = ImageDraw.Draw(mask)

            if shape == "rectangle":
                draw.rectangle([x, y, x + width, y + height], fill=255)
            elif shape in ["circle", "oval"]:
                draw.ellipse([x, y, x + width, y + height], fill=255)

            mask_np = np.array(mask)
            prediction_mask_resized = cv2.bitwise_and(
                prediction_mask_resized,
                mask_np
            )

        # ------------------------------------------------------------
        # 🔹 6. Maskeyi kaydet
        # ------------------------------------------------------------
        mask_filename = f"mask_file{file_id}_{int(time.time())}.png"
        mask_save_path = STATIC_MASKS_DIR / mask_filename

        mask_image = Image.fromarray(
            prediction_mask_resized.astype(np.uint8),
            mode="L"
        )
        mask_image.save(mask_save_path, format="PNG")

        # ------------------------------------------------------------
        # 🔹 7. DB kaydı (file_id ile bağlı)
        # ------------------------------------------------------------
        mask_record = Mask(
            filename=mask_filename,
            file_path=str(mask_save_path),
            file_id=file_record.id
        )

        file_record.status = "segmented"

        db.add(mask_record)
        db.commit()
        db.refresh(mask_record)

        # ------------------------------------------------------------
        # 🔹 8. JSON Response
        # ------------------------------------------------------------
        return JSONResponse(content={
            "mask_id": mask_record.id,
            "mask_url": f"/static/masks/{mask_filename}"
        })

    except Exception as e:
        print("❌ HATA segment:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 1. NIfTI BİLGİSİNİ AL (Kaç kesit var?)
# ============================================================
@router.get("/segment/nifti/{mask_id}/info")
async def get_nifti_info(
    mask_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    mask_record = (
        db.query(Mask)
        .join(Mask.file)
        .join(DBFile.patient)
        .filter(
            Mask.id == mask_id,
            DBFile.patient.has(owner_id=current_user.id)
        )
        .first()
    )

    if not mask_record:
        raise HTTPException(status_code=404, detail="Maske bulunamadı.")

    if not mask_record.file_path.endswith((".nii", ".nii.gz")):
        return {"total_slices": 1, "is_nifti": False}

    try:
        nii = nib.load(mask_record.file_path)
        depth = nii.shape[2]

        return {
            "mask_id": mask_id,
            "total_slices": depth,
            "is_nifti": True
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NIfTI okuma hatası: {str(e)}")

# ============================================================
# 2. BELİRLİ BİR KESİTİ (SLICE) PNG OLARAK GETİR
# ============================================================
@router.get("/segment/nifti/{mask_id}/slice/{slice_index}")
async def get_nifti_slice(
    mask_id: int,
    slice_index: int,
    type: str = "mask",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    mask_record = (
        db.query(Mask)
        .join(Mask.file)
        .join(DBFile.patient)
        .filter(
            Mask.id == mask_id,
            DBFile.patient.has(owner_id=current_user.id)
        )
        .first()
    )

    if not mask_record:
        raise HTTPException(status_code=404, detail="Maske bulunamadı.")

    # ORIGINAL artık Mask içinde değil → File içinden alıyoruz
    target_path = (
        mask_record.file_path
        if type == "mask"
        else mask_record.file.file_path
    )

    if not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail="Dosya diskte yok.")

    try:
        nii = nib.load(target_path)
        data = nii.get_fdata()

        if slice_index >= data.shape[2] or slice_index < 0:
            raise HTTPException(status_code=400, detail="Geçersiz kesit numarası.")

        slice_data = data[:, :, slice_index]

        if np.max(slice_data) > 0:
            slice_data = (slice_data - np.min(slice_data)) / (
                np.max(slice_data) - np.min(slice_data)
            )
            slice_data = (slice_data * 255).astype(np.uint8)
        else:
            slice_data = slice_data.astype(np.uint8)

        img = Image.fromarray(slice_data).convert("L").rotate(90, expand=True)

        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        return StreamingResponse(img_byte_arr, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# 3. DÜZENLENMİŞ KESİTİ 3D DOSYAYA KAYDET (UPDATE)
# ============================================================
@router.post("/segment/nifti/{mask_id}/slice/{slice_index}/update")
async def update_nifti_slice(
    mask_id: int,
    slice_index: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    mask_record = (
        db.query(Mask)
        .join(Mask.file)
        .join(DBFile.patient)
        .filter(
            Mask.id == mask_id,
            DBFile.patient.has(owner_id=current_user.id)
        )
        .first()
    )

    if not mask_record:
        raise HTTPException(status_code=404, detail="Maske bulunamadı.")

    try:
        nii = nib.load(mask_record.file_path)
        data = nii.get_fdata()
        affine = nii.affine

        contents = await file.read()
        new_slice_img = Image.open(io.BytesIO(contents)).convert("L")

        new_slice_img = new_slice_img.rotate(-90, expand=True)
        new_slice_img = new_slice_img.resize(
            (data.shape[0], data.shape[1])
        )

        new_slice_np = np.array(new_slice_img)
        new_slice_np = (new_slice_np > 127).astype(np.float64)

        data[:, :, slice_index] = new_slice_np

        new_nii = nib.Nifti1Image(data, affine)
        nib.save(new_nii, mask_record.file_path)

        return {"status": "success", "slice": slice_index}

    except Exception as e:
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




@router.post("/segment/manual/{file_id}")
async def create_manual_mask(
    file_id: int,
    mask_file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        # ------------------------------------------------------------
        # 🔹 1. File kaydını bul + yetki kontrolü
        # ------------------------------------------------------------
        file_record = (
            db.query(DBFile)
            .join(DBFile.patient)
            .filter(
                DBFile.id == file_id,
                Patient.owner_id == current_user.id
            )
            .first()
        )

        if not file_record:
            raise HTTPException(status_code=404, detail="Dosya bulunamadı.")

        # ------------------------------------------------------------
        # 🔹 2. Maskeyi kaydet (grayscale garanti)
        # ------------------------------------------------------------
        timestamp = int(time.time())
        mask_filename = f"manual_mask_file{file_id}_{timestamp}.png"
        mask_save_path = STATIC_MASKS_DIR / mask_filename

        mask_contents = await mask_file.read()
        mask_image = Image.open(io.BytesIO(mask_contents)).convert("L")
        mask_image.save(mask_save_path, format="PNG")

        # ------------------------------------------------------------
        # 🔹 3. DB kaydı (file_id ile bağlı!)
        # ------------------------------------------------------------
        mask_record = Mask(
            filename=mask_filename,
            file_path=str(mask_save_path),
            file_id=file_record.id
        )

        # File status güncelle (opsiyonel ama önerilir)
        file_record.status = "segmented"

        db.add(mask_record)
        db.commit()
        db.refresh(mask_record)

        # ------------------------------------------------------------
        # 🔹 4. JSON Response
        # ------------------------------------------------------------
        return {
            "mask_id": mask_record.id,
            "mask_url": f"/static/masks/{mask_filename}",
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
    masks = (
        db.query(Mask)
        .join(Mask.file)
        .join(DBFile.patient)
        .filter(DBFile.patient.has(owner_id=current_user.id))
        .all()
    )

    return [
        {
            "mask_id": mask.id,
            "filename": mask.filename,
            "file_id": mask.file_id
        }
        for mask in masks
    ]

@router.delete("/segment/{mask_id}")
def delete_mask(
    mask_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    mask_record = (
        db.query(Mask)
        .join(Mask.file)
        .join(DBFile.patient)
        .filter(
            Mask.id == mask_id,
            DBFile.patient.has(owner_id=current_user.id)
        )
        .first()
    )

    if not mask_record:
        raise HTTPException(status_code=404, detail="Mask bulunamadı.")

    if mask_record.file_path and os.path.exists(mask_record.file_path):
        os.remove(mask_record.file_path)

    db.delete(mask_record)
    db.commit()

    return {"detail": f"Mask (id={mask_id}) silindi."}


@router.put("/segment/{mask_id}")
async def update_mask(
    mask_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    mask_record = (
        db.query(Mask)
        .join(Mask.file)
        .join(DBFile.patient)
        .filter(
            Mask.id == mask_id,
            DBFile.patient.has(owner_id=current_user.id)
        )
        .first()
    )

    if not mask_record:
        raise HTTPException(status_code=404, detail="Mask bulunamadı.")

    file_extension = ".png"
    new_filename = f"mask_{mask_id}_{uuid.uuid4().hex[:8]}{file_extension}"
    save_directory = "static/masks/"
    os.makedirs(save_directory, exist_ok=True)

    file_location = os.path.join(save_directory, new_filename)

    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)

    if os.path.exists(mask_record.file_path):
        os.remove(mask_record.file_path)

    mask_record.filename = new_filename
    mask_record.file_path = file_location

    db.commit()
    db.refresh(mask_record)

    return {
        "mask_id": mask_record.id,
        "filename": mask_record.filename
    }
@router.put("/segment/{mask_id}/replace")
async def replace_mask(
    mask_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    mask_record = (
        db.query(Mask)
        .join(Mask.file)
        .join(DBFile.patient)
        .filter(
            Mask.id == mask_id,
            DBFile.patient.has(owner_id=current_user.id)
        )
        .first()
    )

    if not mask_record:
        raise HTTPException(status_code=404, detail="Mask bulunamadı.")

    if os.path.exists(mask_record.file_path):
        os.remove(mask_record.file_path)

    contents = await file.read()
    new_filename = f"mask_user{current_user.id}_{int(time.time())}.png"
    new_save_path = STATIC_MASKS_DIR / new_filename

    with open(new_save_path, "wb") as f:
        f.write(contents)

    mask_record.filename = new_filename
    mask_record.file_path = str(new_save_path)

    db.commit()
    db.refresh(mask_record)

    return {
        "mask_id": mask_record.id,
        "filename": mask_record.filename
    }
