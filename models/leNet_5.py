import keras
from keras import Sequential

def generateModel(inputShape):
    model = Sequential()

    model.add(keras.layers.Conv2D(6, kernel_size=5, activation='tanh', input_shape=inputShape))
    model.add(keras.layers.AveragePooling2D(pool_size=(2,2), strides=2))

    model.add(keras.layers.Conv2D(16, kernel_size=5, activation='tanh'))
    model.add(keras.layers.AveragePooling2D(pool_size=(2,2), strides=2))

    model.add(keras.layers.Conv2D(32, kernel_size=5, activation='tanh'))
    model.add(keras.layers.AveragePooling2D(pool_size=(2,2), strides=2))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(512, activation='tanh'))
    model.add(keras.layers.Dense(128, activation='tanh'))

    model.add(keras.layers.Dense(45, activation='softmax'))

    return model