import keras
from keras import layers
from keras.models import Model

def residualBlock(x, filters, stride=1):
    shortcut = x

    x = layers.Conv2D(filters, kernel_size=3, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])

    x = layers.Activation('relu')(x)

    return x


def generateSmallResNet(inputShape):

    inputs = keras.Input(shape=inputShape)

    x = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)

    x = residualBlock(x, 32)
    x = residualBlock(x, 32)

    x = residualBlock(x, 64, stride=2)
    x = residualBlock(x, 64)

    x = residualBlock(x, 128, stride=2)
    x = residualBlock(x, 128)

    x = residualBlock(x, 256, stride=2)
    x = residualBlock(x, 256)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(45, activation='softmax')(x)

    model = Model(inputs, outputs)

    return model