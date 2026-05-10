import keras
from keras import Sequential

def generateModel(inputShape):
    model = Sequential()

    model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=inputShape))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

    model.add(keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

    model.add(keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

    model.add(keras.layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

    model.add(keras.layers.GlobalAveragePooling2D())

    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(45, activation='softmax'))

    return model