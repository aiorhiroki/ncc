from ncc.models import Model3D
from ncc.history import save_history
from keras.datasets import mnist
import numpy as np

num_classes = 10
depth = 8
width, height = 28, 28
channel = 1
input_shape = (depth, width, height, channel)
epochs = 30
batch_size = 128

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.asarray(x_train, dtype='float32') / 255.
x_test = np.asarray(x_test, dtype='float32') / 255.
x_train = np.reshape(x_train, (len(x_train),)+input_shape[1:])
x_test = np.reshape(x_test, (len(x_test),)+input_shape[1:])

x_train_3d = np.asarray([x_train for _ in range(depth)]).transpose((1, 0, 2, 3, 4))
x_test_3d = np.asarray([x_test for _ in range(depth)]).transpose((1, 0, 2, 3, 4))


y_train = np.eye(num_classes, dtype='float32')[y_train]
y_test = np.eye(num_classes, dtype='float32')[y_test]


print(x_train_3d.shape, y_train.shape, x_test_3d.shape, y_test.shape)

model = Model3D(input_shape=input_shape, num_classes=num_classes)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(x_train_3d, y_train, validation_data=(x_test_3d, y_test), epochs=epochs, batch_size=batch_size)

save_history(history)
