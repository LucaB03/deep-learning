import keras
from keras import Sequential

def generateModel(inputShape):
    model = Sequential()

    model.add(keras.layers.Conv2D(24, kernel_size=5, padding='same', activation='elu', input_shape=inputShape))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2)))

    model.add(keras.layers.Conv2D(48, kernel_size=3, padding='same', activation='elu'))
    model.add(keras.layers.AveragePooling2D(pool_size=(2,2)))

    model.add(keras.layers.Conv2D(96, kernel_size=3, padding='same', activation='selu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=(2,2)))

    model.add(keras.layers.Conv2D(192, kernel_size=3, padding='same', activation='gelu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.SpatialDropout2D(0.15))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2)))

    model.add(keras.layers.Conv2D(320, kernel_size=3, padding='same', activation='swish'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=(2,2)))

    model.add(keras.layers.GlobalMaxPooling2D())

    model.add(keras.layers.Dense(256, activation='swish'))
    model.add(keras.layers.Dropout(0.4))

    model.add(keras.layers.Dense(45, activation='softmax'))

    return model