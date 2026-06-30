import keras
from keras import Sequential

def generateModel(inputShape):
    model = Sequential()

    model.add(keras.layers.Conv2D(24, 5, padding='same', input_shape=inputShape))
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Conv2D(48, 3, padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Conv2D(96, 3, padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPool2D(2))

    model.add(keras.layers.Conv2D(192, 3, padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.SpatialDropout2D(0.15))

    model.add(keras.layers.Conv2D(256, 3, padding='same'))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPool2D(2))

    model.add(keras.layers.GlobalMaxPooling2D())

    model.add(keras.layers.Dense(256))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Dense(45, activation='softmax'))

    return model