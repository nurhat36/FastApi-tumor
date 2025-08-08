import cv2
from fastapi import APIRouter, UploadFile, File, HTTPException
from starlette.responses import JSONResponse
from PIL import Image
import io
import numpy as np

from app.models.unet_model import model
from app.utils.image_utils import preprocess_image, postprocess_mask, encode_mask_to_base64

router = APIRouter()

@router.post("/segment")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # Görseli aç
        image = Image.open(io.BytesIO(contents)).convert("L")  # Grayscale
        image = image.resize((128, 128))  # Modelin beklediği boyut

        image_np = np.array(image, dtype=np.float32) / 255.0  # Normalize
        image_np = np.expand_dims(image_np, axis=-1)  # (128,128) -> (128,128,1)
        image_np = np.expand_dims(image_np, axis=0)   # (128,128,1) -> (1,128,128,1)

        # Tahmin yap
        prediction = model.predict(image_np)[0]
        prediction_mask = (prediction > 0.5).astype(np.uint8)

        # Maske resize (128x128 -> orijinal boyut)
        prediction_mask = cv2.resize(prediction_mask, image.size).tolist()

        return JSONResponse(content={"mask": prediction_mask})

    except Exception as e:
        print("HATA:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
