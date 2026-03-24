# app/routers/segment.py
import io, time
import os
import shutil

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends

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

def segment_volume_nifti_with_roi(file_bytes, original_filename, current_user, axis, x, y, width, height, shape):
    ext = ".nii.gz" if original_filename.endswith(".nii.gz") else ".nii"
    temp_filename = f"temp_{uuid.uuid4().hex}{ext}"
    temp_path = STATIC_MASKS_DIR / temp_filename

    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    nii = nib.load(str(temp_path))
    volume = nii.get_fdata()
    affine = nii.affine

    # Çıktı için boş bir 3D hacim oluştur (Tümü 0)
    mask_volume = np.zeros_like(volume, dtype=np.float64)

    # Hangi eksende döneceğiz?
    if axis == "sagittal":
        loop_range = volume.shape[0]
    elif axis == "coronal":
        loop_range = volume.shape[1]
    else:  # Varsayılan: axial
        loop_range = volume.shape[2]

    print(f"🧠 Segmentasyon Başladı: {loop_range} kesit {axis} ekseninde analiz edilecek...")

    # BÜTÜN DİLİMLERİ (KESİTLERİ) DÖN
    for i in range(loop_range):
        # 1. Kesiti al
        if axis == "sagittal":
            slice_img = volume[i, :, :]
        elif axis == "coronal":
            slice_img = volume[:, i, :]
        else:
            slice_img = volume[:, :, i]

        # 2. Normalize et
        if np.max(slice_img) > 0:
            slice_norm = (slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img))
            slice_norm = (slice_norm * 255).astype(np.uint8)
        else:
            slice_norm = slice_img.astype(np.uint8)

        # 3. Flutter'daki açıyla eşleşmesi için 90 derece döndür
        pil_slice = Image.fromarray(slice_norm).convert("L").rotate(90, expand=True)
        original_size = pil_slice.size

        # 4. Modeli çalıştır
        IMG_SIZE = 128
        image_resized = pil_slice.resize((IMG_SIZE, IMG_SIZE))
        image_np = np.array(image_resized, dtype=np.float32) / 255.0
        image_np = np.expand_dims(image_np, axis=-1)
        image_np = np.expand_dims(image_np, axis=0)

        prediction = model.predict(image_np, verbose=0)[0]  # verbose=0 log kirliliğini önler

        if prediction.ndim == 3 and prediction.shape[-1] > 1:
            prediction = prediction[..., 0]

        prediction_mask = (prediction > 0.5).astype(np.uint8) * 255
        if prediction_mask.ndim == 3 and prediction_mask.shape[-1] == 1:
            prediction_mask = np.squeeze(prediction_mask, axis=-1)

        prediction_mask_resized = cv2.resize(prediction_mask, original_size)

        # 5. KULLANICININ SEÇTİĞİ ALANI (ROI) UYGULA (Sadece o alanı bırak, gerisini sil)
        if width > 0 and height > 0:
            roi_mask = Image.new("L", original_size, 0)
            draw = ImageDraw.Draw(roi_mask)
            if shape == "rectangle":
                draw.rectangle([x, y, x + width, y + height], fill=255)
            elif shape in ["circle", "oval"]:
                draw.ellipse([x, y, x + width, y + height], fill=255)

            roi_mask_np = np.array(roi_mask)
            prediction_mask_resized = cv2.bitwise_and(prediction_mask_resized, roi_mask_np)

        # 6. Kesiti eski NIfTI açısına döndür (-90)
        final_mask_pil = Image.fromarray(prediction_mask_resized).rotate(-90, expand=True)
        final_mask_np = np.array(final_mask_pil).astype(np.float64) / 255.0

        # 7. Oluşan maskeyi 3D hacimdeki yerine koy
        if axis == "sagittal":
            mask_volume[i, :, :] = final_mask_np
        elif axis == "coronal":
            mask_volume[:, i, :] = final_mask_np
        else:
            mask_volume[:, :, i] = final_mask_np

    # Tüm dilimler bitti, 3D dosyayı kaydet
    timestamp = int(time.time())
    mask_filename = f"mask_volume_user{current_user.id}_{timestamp}.nii.gz"
    mask_save_path = STATIC_MASKS_DIR / mask_filename

    mask_nifti = nib.Nifti1Image(mask_volume, affine)
    nib.save(mask_nifti, str(mask_save_path))

    os.remove(temp_path)
    return mask_filename
# ============================================================
# 🎯 Segmentasyon Endpoint (YENİ MİMARİ)
# ============================================================
def segment_volume_with_3d_roi(file_bytes, original_filename, current_user,
                               ax_x, ax_y, ax_w, ax_h,
                               cor_x, cor_y, cor_w, cor_h,
                               sag_x, sag_y, sag_w, sag_h, shape):
    ext = ".nii.gz" if original_filename.endswith(".nii.gz") else ".nii"
    temp_filename = f"temp_{uuid.uuid4().hex}{ext}"
    temp_path = STATIC_MASKS_DIR / temp_filename

    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    nii = nib.load(str(temp_path))
    volume = nii.get_fdata()
    affine = nii.affine

    dimX, dimY, dimZ = volume.shape

    # 1. BÜTÜN 3D HACMİ KAPSAYAN BİR KALIP OLUŞTUR (İçi full 1 dolu)
    roi_3d = np.ones((dimX, dimY, dimZ), dtype=np.float64)

    # 2. AXIAL (Üstten) Sınırlandırma
    if ax_w > 0 and ax_h > 0:
        mask2d = Image.new("L", (dimX, dimY), 0)
        draw = ImageDraw.Draw(mask2d)
        if shape == "rectangle":
            draw.rectangle([ax_x, ax_y, ax_x + ax_w, ax_y + ax_h], fill=255)
        elif shape in ["circle", "oval"]:
            draw.ellipse([ax_x, ax_y, ax_x + ax_w, ax_y + ax_h], fill=255)
        mask2d_np = np.array(mask2d.rotate(-90, expand=True)) / 255.0
        for z in range(dimZ): roi_3d[:, :, z] *= mask2d_np

    # 3. CORONAL (Önden) Sınırlandırma
    if cor_w > 0 and cor_h > 0:
        mask2d = Image.new("L", (dimX, dimZ), 0)
        draw = ImageDraw.Draw(mask2d)
        if shape == "rectangle":
            draw.rectangle([cor_x, cor_y, cor_x + cor_w, cor_y + cor_h], fill=255)
        elif shape in ["circle", "oval"]:
            draw.ellipse([cor_x, cor_y, cor_x + cor_w, cor_y + cor_h], fill=255)
        mask2d_np = np.array(mask2d.rotate(-90, expand=True)) / 255.0
        for y in range(dimY): roi_3d[:, y, :] *= mask2d_np

    # 4. SAGITTAL (Yandan) Sınırlandırma
    if sag_w > 0 and sag_h > 0:
        mask2d = Image.new("L", (dimY, dimZ), 0)
        draw = ImageDraw.Draw(mask2d)
        if shape == "rectangle":
            draw.rectangle([sag_x, sag_y, sag_x + sag_w, sag_y + sag_h], fill=255)
        elif shape in ["circle", "oval"]:
            draw.ellipse([sag_x, sag_y, sag_x + sag_w, sag_y + sag_h], fill=255)
        mask2d_np = np.array(mask2d.rotate(-90, expand=True)) / 255.0
        for x in range(dimX): roi_3d[x, :, :] *= mask2d_np

    # 5. YAPAY ZEKA ANALİZİ (Artık sadece Axial döngüsü yeterli)
    pred_volume = np.zeros_like(volume, dtype=np.float64)

    print("🧠 Segmentasyon Başladı. Optimizasyon devrede...")
    for z in range(dimZ):
        # MÜTHİŞ OPTİMİZASYON: Eğer bu dilimde ROI tamamen sıfırsa (kullanıcı alanı dışında kalıyorsa) Yapay Zekayı HİÇ ÇALIŞTIRMA! (Çok Hızlandırır)
        if np.max(roi_3d[:, :, z]) == 0:
            continue

        slice_img = volume[:, :, z]
        if np.max(slice_img) > 0:
            slice_norm = (slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img))
            slice_norm = (slice_norm * 255).astype(np.uint8)
        else:
            slice_norm = slice_img.astype(np.uint8)

        pil_slice = Image.fromarray(slice_norm).convert("L").rotate(90, expand=True)
        original_size = pil_slice.size

        IMG_SIZE = 128
        image_resized = pil_slice.resize((IMG_SIZE, IMG_SIZE))
        image_np = np.array(image_resized, dtype=np.float32) / 255.0
        image_np = np.expand_dims(image_np, axis=-1)
        image_np = np.expand_dims(image_np, axis=0)

        prediction = model.predict(image_np, verbose=0)[0]
        if prediction.ndim == 3 and prediction.shape[-1] > 1:
            prediction = prediction[..., 0]

        prediction_mask = (prediction > 0.5).astype(np.uint8) * 255
        if prediction_mask.ndim == 3 and prediction_mask.shape[-1] == 1:
            prediction_mask = np.squeeze(prediction_mask, axis=-1)

        prediction_mask_resized = cv2.resize(prediction_mask, original_size)
        final_mask_pil = Image.fromarray(prediction_mask_resized).rotate(-90, expand=True)
        pred_volume[:, :, z] = np.array(final_mask_pil).astype(np.float64) / 255.0

    # 6. YAPAY ZEKANIN SONUÇLARINI, 3 BOYUTLU KALIBIMIZLA KESİŞTİR (Filtrele)
    final_volume = pred_volume * roi_3d

    # 7. KAYDET
    timestamp = int(time.time())
    mask_filename = f"mask_volume_user{current_user.id}_{timestamp}.nii.gz"
    mask_save_path = STATIC_MASKS_DIR / mask_filename

    mask_nifti = nib.Nifti1Image(final_volume, affine)
    nib.save(mask_nifti, str(mask_save_path))

    os.remove(temp_path)
    return mask_filename


@router.post("/segment/{file_id}")
async def predict_image(
        file_id: int,
        ax_x: float = Form(0), ax_y: float = Form(0), ax_w: float = Form(0), ax_h: float = Form(0),
        cor_x: float = Form(0), cor_y: float = Form(0), cor_w: float = Form(0), cor_h: float = Form(0),
        sag_x: float = Form(0), sag_y: float = Form(0), sag_w: float = Form(0), sag_h: float = Form(0),
        shape: str = Form("rectangle"),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    try:
        file_record = db.query(DBFile).join(DBFile.patient).filter(DBFile.id == file_id,
                                                                   Patient.owner_id == current_user.id).first()
        if not file_record or not os.path.exists(file_record.file_path):
            raise HTTPException(status_code=404, detail="Dosya bulunamadı.")

        with open(file_record.file_path, "rb") as f:
            contents = f.read()

        # 🔥 NIFTI VOLUME SEGMENTATION
        if file_record.file_path.endswith((".nii", ".nii.gz")):
            mask_filename = segment_volume_with_3d_roi(
                contents, file_record.filename, current_user,
                ax_x, ax_y, ax_w, ax_h,
                cor_x, cor_y, cor_w, cor_h,
                sag_x, sag_y, sag_w, sag_h, shape
            )

            mask_save_path = STATIC_MASKS_DIR / mask_filename
            mask_record = Mask(filename=mask_filename, file_path=str(mask_save_path), file_id=file_record.id)

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
        # 🔹 5. Kullanıcı ROI uygulaması (🔥 DÜZELTİLEN KISIM 🔥)
        # ------------------------------------------------------------
        # Normal 2D resimlerde çizimi 'axial' eksenine kaydettiğimiz için
        # ax_w, ax_h, ax_x, ax_y değişkenlerini kullanıyoruz.
        if ax_w > 0 and ax_h > 0:
            mask = Image.new("L", original_size, 0)
            draw = ImageDraw.Draw(mask)

            if shape == "rectangle":
                draw.rectangle([ax_x, ax_y, ax_x + ax_w, ax_y + ax_h], fill=255)
            elif shape in ["circle", "oval"]:
                draw.ellipse([ax_x, ax_y, ax_x + ax_w, ax_y + ax_h], fill=255)

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
# ============================================================
# 2. BELİRLİ BİR KESİTİ (SLICE) PNG OLARAK GETİR
# ============================================================
@router.get("/segment/nifti/{mask_id}/slice/{slice_index}")
async def get_nifti_slice(
        mask_id: int,
        slice_index: int,
        type: str = "mask",
        axis: str = "axial",  # 🔥 YENİ: Hangi eksenden bakıldığını tutan parametre (Varsayılan: axial)
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

        # 🔥 YENİ: EKSENE GÖRE KESİTİ (DİLİMİ) AL
        if axis == "sagittal":
            if slice_index >= data.shape[0] or slice_index < 0:
                raise HTTPException(status_code=400, detail="Geçersiz Sagittal kesit numarası.")
            slice_data = data[slice_index, :, :]

        elif axis == "coronal":
            if slice_index >= data.shape[1] or slice_index < 0:
                raise HTTPException(status_code=400, detail="Geçersiz Coronal kesit numarası.")
            slice_data = data[:, slice_index, :]

        else:  # Varsayılan olarak "axial" (Üstten)
            if slice_index >= data.shape[2] or slice_index < 0:
                raise HTTPException(status_code=400, detail="Geçersiz Axial kesit numarası.")
            slice_data = data[:, :, slice_index]

        # --------------------------------------------------------
        # Görüntü Normalizasyonu ve Döndürme (Mevcut kodunuz)
        # --------------------------------------------------------
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


# ============================================================
# 4. HAM (HENÜZ ANALİZ EDİLMEMİŞ) NIfTI BİLGİSİNİ AL
# ============================================================
@router.get("/files/nifti/{file_id}/info")
async def get_raw_nifti_info(
        file_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    file_record = db.query(DBFile).join(Patient).filter(DBFile.id == file_id,
                                                        Patient.owner_id == current_user.id).first()
    if not file_record or not os.path.exists(file_record.file_path):
        raise HTTPException(status_code=404, detail="Dosya bulunamadı")

    nii = nib.load(file_record.file_path)
    shape = nii.shape

    # shape[0] -> Sagittal (X ekseni)
    # shape[1] -> Coronal (Y ekseni)
    # shape[2] -> Axial (Z ekseni)
    return {
        "total_slices": shape[2],  # Geriye dönük uyumluluk için (eski kod bozulmasın diye)
        "axial_slices": shape[2],  # Üstten (Z)
        "coronal_slices": shape[1],  # Önden (Y)
        "sagittal_slices": shape[0]  # Yandan (X)
    }


# ============================================================
# 5. HAM NIfTI KESİTİNİ PNG OLARAK GETİR (MASKESİZ)
# ============================================================
@router.get("/files/nifti/{file_id}/slice/{slice_index}")
async def get_raw_nifti_slice(
        file_id: int,
        slice_index: int,
        axis: str = "axial",  # YENİ: Flutter'dan gelecek eksen parametresi (Varsayılan: axial)
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    file_record = db.query(DBFile).join(Patient).filter(DBFile.id == file_id,
                                                        Patient.owner_id == current_user.id).first()
    if not file_record or not os.path.exists(file_record.file_path):
        raise HTTPException(status_code=404, detail="Dosya bulunamadı")

    nii = nib.load(file_record.file_path)
    data = nii.get_fdata()

    # --------------------------------------------------------
    # EKSENE GÖRE KESİTİ (DİLİMİ) AL VE KONTROL ET
    # --------------------------------------------------------
    if axis == "sagittal":
        if slice_index >= data.shape[0] or slice_index < 0:
            raise HTTPException(status_code=400, detail="Geçersiz Sagittal kesit")
        slice_data = data[slice_index, :, :]

    elif axis == "coronal":
        if slice_index >= data.shape[1] or slice_index < 0:
            raise HTTPException(status_code=400, detail="Geçersiz Coronal kesit")
        slice_data = data[:, slice_index, :]

    else:  # Varsayılan olarak "axial" kabul et
        if slice_index >= data.shape[2] or slice_index < 0:
            raise HTTPException(status_code=400, detail="Geçersiz Axial kesit")
        slice_data = data[:, :, slice_index]

    # --------------------------------------------------------
    # GÖRÜNTÜYÜ NORMALIZE ET (0-255 ARASI) VE PNG'YE ÇEVİR
    # --------------------------------------------------------
    if np.max(slice_data) > 0:
        slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))
        slice_data = (slice_data * 255).astype(np.uint8)
    else:
        slice_data = slice_data.astype(np.uint8)

    # NIfTI dosyaları matris olarak okunurken genelde 90 derece yatık gelir.
    # Mevcut kodunuzdaki rotate(90) mantığını koruduk.
    img = Image.fromarray(slice_data).convert("L").rotate(90, expand=True)

    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/png")