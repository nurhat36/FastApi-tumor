from fastapi import APIRouter, UploadFile, File, HTTPException
from starlette.responses import JSONResponse
from PIL import Image
import io

from app.models.unet_model import model
from app.utils.image_utils import preprocess_image, postprocess_mask, encode_mask_to_base64

router = APIRouter()

@router.post("/segment")
async def segment(image: UploadFile = File(...)):
    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Sadece jpeg veya png dosyaları kabul edilir")

    img_bytes = await image.read()
    image_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    input_data = preprocess_image(image_pil)
    prediction = model.predict(input_data)

    mask = postprocess_mask(prediction)
    encoded_mask = encode_mask_to_base64(mask)

    return JSONResponse(content={"mask": encoded_mask})
