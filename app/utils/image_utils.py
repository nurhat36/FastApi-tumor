import numpy as np
from PIL import Image
import cv2
import base64

def preprocess_image(image: Image.Image):
    image = image.resize((256, 256))  # Model eğitim boyutuna göre ayarla
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def postprocess_mask(mask):
    mask = (mask[0, :, :, 0] > 0.5).astype(np.uint8) * 255
    return mask

def encode_mask_to_base64(mask):
    _, buffer = cv2.imencode('.png', mask)
    return base64.b64encode(buffer).decode('utf-8')
