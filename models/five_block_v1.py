import keras
from keras import Sequential

def generateModel(inputShape):
    model = Sequential()

    model.add(keras.layers.Conv2D(16, kernel_size=3, padding='same', activation='relu', input_shape=inputShape))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2)))

    model.add(keras.layers.Conv2D(40, kernel_size=5, padding='same', activation='leaky_relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.SpatialDropout2D(0.1))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2)))

    model.add(keras.layers.Conv2D(80, kernel_size=3, padding='same', activation='tanh'))
    model.add(keras.layers.AveragePooling2D(pool_size=(2,2)))

    model.add(keras.layers.Conv2D(160, kernel_size=3, padding='same', activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2)))

    model.add(keras.layers.Conv2D(288, kernel_size=3, padding='same', activation='gelu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.SpatialDropout2D(0.2))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2)))

    model.add(keras.layers.GlobalAveragePooling2D())

    model.add(keras.layers.Dense(384, activation='selu'))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(45, activation='softmax'))

    return model