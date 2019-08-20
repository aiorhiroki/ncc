from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.xception import Xception
from keras.models import Model


def xception(nb_classes, width=299, height=299):
    base_model = Xception(input_shape=(width, height, 3), weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(base_model.input, predictions)

    return model
