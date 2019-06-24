# coding: utf-8

from PIL import Image, ImageEnhance, ImageOps, ImageFilter

import numpy as np
from scipy.misc import imresize


def random_crop(image, crop_size, mask):
    h, w, _ = image.shape

    if h - crop_size[0] == 0:
        top = 0
    else:
        top = np.random.randint(0, h - crop_size[0])

    if w - crop_size[1] == 0:
        left = 0
    else:
        left = np.random.randint(0, w - crop_size[1])

    bottom = top + crop_size[0]
    right = left + crop_size[1]

    # 決めたtop, bottom, left, rightを使って画像を抜き出す
    image = image[top:bottom, left:right, :]

    if mask:
        mask = mask[top:bottom, left:right, :]

    return image, mask


def scale_augmentation(image, crop_size, mask):
    h, w, _ = image.shape
    scale_size_w = int(w*np.random.randint(1000, 1200)/1000)  # 1.0 ~ 1.2
    scale_size_h = int(h*np.random.randint(1000, 1200)/1000)  # 1.0 ~ 1.2
    image = imresize(image, (scale_size_h, scale_size_w))
    image, mask = random_crop(image, crop_size, mask)
    return image, mask


def data_augmentation(image_array, crop_size=(256, 256), mask_array=None):
    SATURATION = np.random.rand()  # 0.0 ~ 1.0
    CONTRAST = np.random.randint(500, 1000)/1000  # 0.5 ~ 1.0
    BRIGHTNESS = np.random.randint(500, 1000)/1000  # 0.5 ~ 1.0
    SHARPNESS = np.random.randint(0, 2000)/1000  # 0.0 ~ 2.0
    FLIP = np.random.choice([True, False])  # True or False
    MIRROR = np.random.choice([True, False])  # True or False
    BLUR = np.random.randint(0, 1000)/1000  # 0.0 ~ 1.0

    img = Image.fromarray(np.uint8(image_array))
    if mask_array:
        mask_pil = Image.fromarray(np.uint8(mask_array))
    else:
        mask_pil = None

    # 彩度を変える
    saturation_converter = ImageEnhance.Color(img)
    img = saturation_converter.enhance(SATURATION)

    # コントラストを変える
    contrast_converter = ImageEnhance.Contrast(img)
    img = contrast_converter.enhance(CONTRAST)

    # 明度を変える
    brightness_converter = ImageEnhance.Brightness(img)
    img = brightness_converter.enhance(BRIGHTNESS)

    # シャープネスを変える
    sharpness_converter = ImageEnhance.Sharpness(img)
    img = sharpness_converter.enhance(SHARPNESS)

    if FLIP:
        img = ImageOps.flip(img)  # 上下反転
        if mask_pil:
            mask_pil = ImageOps.flip(mask_pil)

    if MIRROR:
        img = ImageOps.mirror(img)    # 左右反転
        if mask_pil:
            mask_pil = ImageOps.mirror(mask_pil)

    img = img.filter(ImageFilter.GaussianBlur(BLUR))  # ガウシアンブラー

    array = np.asarray(img)
    if mask_pil:
        mask = np.asarray(mask_pil)
    else:
        mask = None

    array, mask = scale_augmentation(array, crop_size, mask)
    return array, mask
