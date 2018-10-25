from ncc.models import Model2D
from ncc.history import save_history
from ncc.preprocessing import preprocess_input
from ncc.metrics import show_matrix, roc

from keras.datasets import mnist

import numpy as np

# parameters
num_classes = 10
input_shape = (28, 28, 1)
epochs = 1
batch_size = 128

# prepare data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)
x_train, y_train = preprocess_input(x_train, y_train)
x_test, y_test = preprocess_input(x_test, y_test)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# build and train model
model = Model2D(input_shape=input_shape, num_classes=num_classes)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc'])
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size)

# save history and model
save_history(history)
model.save_weights('cnn_2d_model.h5')

# confusion matrix
y_prediction = model.predict(x_test)
print(y_prediction.shape)
y_prediction_cls = np.argmax(y_prediction, axis=1)  # from one hot to class index
y_test_cls = np.argmax(y_test, axis=1)  # from one hot to class index
show_matrix(y_test_cls, y_prediction_cls, [i for i in range(10)])
roc(y_test, y_prediction, 10)
