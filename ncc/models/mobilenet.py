import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.models import Model

import pretrainedmodels


def mobilenet(nb_classes, width=244, height=244, framework="tensorflow"):
    if framework == "tensorflow":
        with tf.device("/cpu:0"):
            base_model = MobileNet(
                input_shape=(width, height, 3),
                weights='imagenet',
                include_top=False
            )
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            predictions = Dense(nb_classes, activation='softmax')(x)
            model = Model(base_model.input, predictions)

    elif framework == "pytorch":
        model = pretrainedmodels.__dict__['mobilenet'](
            num_classes=nb_classes, pretrained='imagenet'
        )

    return model
