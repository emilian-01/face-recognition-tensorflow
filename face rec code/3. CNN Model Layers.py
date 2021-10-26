# Import Tensorflow Libraries
from tensorflow.keras import layers
from tensorflow.keras import Model
import matplotlib.pyplot as plt

img_input = layers.Input(shape=(150, 150, 3))

# 2D Convolution Layer with 64 filters of dimension 3x3 and ReLU  activation algorithm
x = layers.Conv2D(64, 3, activation='relu')(img_input)
# 2D Max Pooling Layer
x = layers.MaxPooling2D(2)(x)

# 2D Convolution Layer with 128 filters of dimension 3x3 and ReLU  activation algorithm
x = layers.Conv2D(128, 3, activation='relu')(x)
# 2D Max Pooling Layer
x = layers.MaxPooling2D(2)(x)

# 2D Convolution Layer with 256 filters of dimension 3x3 and ReLU  activation algorithm
x = layers.Conv2D(256, 3, activation='relu')(x)
# 2D Max Pooling Layer
x = layers.MaxPooling2D(2)(x)

# 2D Convolution Layer with 512 filters of dimension 3x3 and ReLU  activation algorithm
x = layers.Conv2D(512, 3, activation='relu')(x)
# 2D Max Pooling Layer
x = layers.MaxPooling2D(2)(x)

# 2D Convolution Layer with 512 filters of dimension 3x3 and ReLU  activation algorithm
x = layers.Conv2D(512, 3, activation='relu')(x)
# 2D Max Pooling Layer
x = layers.Flatten()(x)

# Fully Connected Layers and ReLU activation algorithm
x = layers.Dense(4096, activation='relu')(x)
x = layers.Dense(4096, activation='relu')(x)
x = layers.Dense(1000, activation='relu')(x)

# Dropout Layer for optimization
x = layers.Dropout(0.5)(x)

# Fully Connected Layers and sigmoid activation algorithm
output = layers.Dense(1, activation='sigmoid')(x)

model = Model(img_input, output)

model.summary()