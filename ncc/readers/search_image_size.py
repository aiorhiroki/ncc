from glob import glob
from collections import Counter
import numpy as np
import pandas as pd
import re
from PIL import Image


def search_from_dir(target_dir):
    files = []
    for image_suffix in ['jpg', 'png', 'jpeg']:
        image_paths = target_dir + '/*/*.' + image_suffix  # target_dir/train(test)/class_name/*.jpg
        files += glob(image_paths)
    return median_size_from_files(files)


def search_from_annotation(annotation_file):
    annotation_df = pd.read_csv(annotation_file).values.T
    files = annotation_df[0]
    class_index = annotation_df[1]

    class_names = np.unique(class_index)

    return median_size_from_files(files), class_names


def median_size_from_files(files):
    if len(files) > 10000:
        files = files[:10000]  # ignore large image files

    height_list, width_list, channel_list = [], [], []
    for file in files:
        if re.search('.jpg|jpeg|bmp|png|JPG|JPEG|BMP|PNG|', file):
            image = Image.open(file)
            width, height = image.size[:2]
            channel = 1 if len(np.array(image).shape) == 2 else 3
            height_list.append(height)
            width_list.append(width)
            channel_list.append(channel)
    height_median = np.median(height_list).astype('int')
    width_median = np.median(width_list).astype('int')
    counter = Counter(channel_list)
    channel_most = counter.most_common(1)[0][0]

    return height_median, width_median, channel_most
