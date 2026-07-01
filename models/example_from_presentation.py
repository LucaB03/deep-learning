import keras
from keras import Sequential

def generateModel(inputShape):
    model = Sequential()
    # 1st conv layer
    model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=inputShape))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
    # 2nd conv layer
    model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
    # fully connected layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=500, activation='relu'))
    model.add(keras.layers.Dense(units=45, activation='softmax'))
    return model