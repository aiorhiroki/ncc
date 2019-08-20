import numpy as np
import pandas as pd
import re

from ncc.preprocessing import preprocess_input

from keras.preprocessing.image import load_img, img_to_array


def generate_arrays_from_annotation(annotation_file, batch_size, nb_classes, target_size, dimension=2, nb_frames=32, training=True):

    annotation_data = pd.read_csv(annotation_file).values
    if training:
        np.random.shuffle(annotation_data)

    while True:

        x, y = [], []
        i = 0

        for image_path, label_index in annotation_data:
            image = _set_input_data(image_path, target_size, dimension, nb_frames)
            y.append(label_index)
            x.append(image)
            i += 1
            if i == batch_size:
                yield (preprocess_input(np.array(x)), np.eye(nb_classes)[np.array(y)])
                i = 0
                x, y = [], []


def _set_input_data(image_path, target_size, dimension, nb_frames):
    # color order: (R, G, B)
    if dimension == 2:
        image = load_img(image_path, target_size=target_size)

        array = img_to_array(image)

    # stack sequence image with index (e.g. from image-4.jpg to image-36.jpg)
    elif dimension == 3:
        array = []
        first_frame_index = int(re.findall('(\d+)', image_path)[-1])

        for frame_index in range(first_frame_index, first_frame_index + nb_frames):
            sequence_path = image_path.replace(str(first_frame_index), str(frame_index))
            frame = load_img(sequence_path, target_size=target_size)
            array.append(img_to_array(frame))

        array = np.asarray(array)

    else:
        raise ValueError('invalid dimension arg. dimension should be 2 or 3')

    return array
