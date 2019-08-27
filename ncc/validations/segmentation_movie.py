import cv2
import numpy as np
from PIL import Image
from ncc.video import FPS
from ncc.models import xception, Deeplabv3
from ncc.utils import palette
from segmentation_models.metrics import iou_score
from segmentation_models.losses import cce_dice_loss


def build_model(input_shape, classes, weight_path):
    model = Deeplabv3(
            input_shape=input_shape,
            classes=classes,
            backbone='xception'
        )
    model.load_weights(weight_path)
    model.compile(optimizer='adam', loss=cce_dice_loss,
                          metrics=[iou_score])

    return model


def get_mask(image, model):
    pred = model.predict(image)
    mask = np.argmax(pred, axis=-1)
    mask = mask.astype('uint8')

    return mask, pred


def apply_mask(frame, mask):
    mask = Image.fromarray(np.uint8(mask), mode="P")
    mask.putpalette(palette.palettes)
    mask = mask.convert('RGB')
    mask = np.array(mask)
    seg_image = cv2.add(frame,mask)

    return seg_image


def segmentation_movie(weight_path, video_path, classes=2, input_shape=(256, 512, 3), start_frame=1, end_frame=None):
    '''
    Show segmentation movie
    '''
    model = build_model(input_shape, classes, weight_path)

    video = cv2.VideoCapture(video_path)
    video.set(1, start_frame)
    fps = FPS()

    if not end_frame:
        end_frame = video.get(7) # total frame count

    frame_num = 0
    while frame_num != end_frame:
        ret, frame = video.read()

        frame = cv2.resize(frame, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_NEAREST)
        img = np.expand_dims(frame, axis=0).astype('float32')/255.

        mask, pred = get_mask(img,model)
        mask = mask.reshape(input_shape[:2])
        seg_image = apply_mask(frame, mask)
        frame_num += 1

        cv2.imshow('segmentation', seg_image)
        if (cv2.waitKey(1) & 0xFF == ord('q')) or (ret == False):
            break

    video.release()
    cv2.destroyAllWindows()
