# Model class API
## Model2D
```python
Model2D(input_shape, num_classes, include_top=True)
```
### arguments
- input_shape: (height, width, channel)
- num_classes: number of output class
- include_top: include fully connected layers (defualt: True)
### returns
- Model
`Model2D` build CNN based model automatically. 
```python
from ncc.models import Model2D

# You can use any input shape.
model = Model2D(input_shape=(64, 128, 3), num_classes=10, include_top=True)

model.compile(...)
model.fit(...)
```
If you build your own Fully Connected network, `include_top=False`
```python
from ncc.models import Model2D

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

base_model = Model2D(input_shape=(64, 128, 3), num_classes=10, include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(100, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(...)
model.fit(...)
```

## Model3D
```python
Model3D(input_shape, num_classes, include_top=True)
```
### arguments
- input_shape: (depth, height, width, channel)
- num_classes: number of output class
- include_top: include fully connected layers (defualt: True)
### returns
- Model

`Model3D` build CNN based model automatically. 
```python
from ncc.models import Model3D

# You can use any input shape.
model = Model3D(input_shape=(16, 64, 128, 3), num_classes=10, include_top=True)

model.compile(...)
model.fit(...)
```

## Unet
```python
Unet(input_shape, output_channel_count)
```
### arguments
- input_shape: (height, width, channel)
- output_channel_count: number of output channel(or class)
### returns
- Model
- input shape: it may change, if not optimal for Unet.
already compiled with (optimizer is `sgd`, loss and metrics is base on `dice coefficient`)
```python
from ncc.models import Unet

model, input_shape = Unet(input_shape=(640, 1080, 1), output_channel_count=1)

# (you may should change input shape)

model.compile(...)
model.fit(...)
```

if you want change your model configuration
```python
from ncc.models import Unet
from keras import optimizers
from keras import losses

sgd = optimizers.SGD(lr=0.5)
mse = losses.mean_squared_error

model, input_shape = Unet(input_shape=(640, 1080, 1), output_channel_count=1)
model.optimizer = sgd
model.loss = mse

# (you may should change input shape)

model.compile(...)
model.fit(...)
```
