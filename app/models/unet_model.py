from keras.models import load_model
from app.models.metrics import dice_loss, dice_coef, iou_metric
from app.models.advanced_unet import build_advanced_unet

model = build_advanced_unet()

model = load_model('app/static/unet_model.h5', custom_objects={
    'dice_loss': dice_loss,
    'dice_coef': dice_coef,
    'iou_metric': iou_metric
}, compile=False)
