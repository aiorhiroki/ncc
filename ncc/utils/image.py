import colorsys
from .palette import palettes
import numpy as np
import random
from PIL import Image


def random_colors(N, bright=True, scale=True, shuffle=False):
    """ Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    if scale:
        colors = tuple(np.array(colors)*255)
    if shuffle:
        random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """ Apply the given mask to the image.
    image: (height, width, channel)
    mask: (height, width)
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def convert_to_palette(numpy_image):
    pil_palette = Image.fromarray(np.uint8(numpy_image), mode="P")
    pil_palette.putpalette(palettes)
    return pil_palette


def change_color_palettes(image_files, colors):
    for image_file in image_files:
        pil_img = Image.open(image_file)
        numpy_image = np.array(pil_img, dtype=np.uint8)
        for i, color in enumerate(colors):
            numpy_image[numpy_image == color] = i
        pil_palette = convert_to_palette(numpy_image)
        pil_palette.save(image_file)


def concat_images(im1, im2, palette, mode):
    if mode == "P":
        assert palette is not None
        dst = Image.new("P", (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        dst.putpalette(palette)
    elif mode == "RGB":
        dst = Image.new("RGB", (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
    else:
        raise NotImplementedError

    return dst


def cast_to_pil(ndarray, palette, index_void=None):
    # index_void: 境界線のindexで学習・可視化の際は背景色と同じにする。
    assert len(ndarray.shape) == 3
    res = np.argmax(ndarray, axis=2)
    if index_void is not None:
        res = np.where(res == index_void, 0, res)
    image = Image.fromarray(np.uint8(res), mode="P")
    image.putpalette(palette)
    return image


def get_imageset(
    image_in_np,
    image_out_np,
    image_gt_np,
    palette,
    index_void=None
):
    # 3つの画像(in, out, gt)をくっつけます。
    image_out = cast_to_pil(
        image_out_np, palette, index_void
    )
    image_tc = cast_to_pil(
        image_gt_np, palette, index_void
    )
    image_merged = concat_images(
        image_out, image_tc, palette, "P"
    ).convert("RGB")
    image_in_pil = Image.fromarray(
        np.uint8(image_in_np * 255), mode="RGB"
    )
    image_result = concat_images(
        image_in_pil, image_merged, None, "RGB"
    )
    return image_result
