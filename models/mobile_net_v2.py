import keras
import tensorflow as tf
from keras import Sequential, layers, models
from models.lib import get_augmentation_layer

def generate_model(input_shape=(256, 256, 3), num_classes=45):
    def inverted_residual(x, expansion, filters, stride):
        shortcut = x
        in_channels = x.shape[-1]
        x = layers.Conv2D(expansion * in_channels, (1, 1), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(max_value=6.0)(x)
        x = layers.DepthwiseConv2D((3, 3), strides=stride, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(max_value=6.0)(x)
        x = layers.Conv2D(filters, (1, 1), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        if stride == 1 and in_channels == filters:
            x = layers.add([x, shortcut])
        return x

    inputs = layers.Input(shape=input_shape)
    x = get_augmentation_layer()(inputs)

    x = layers.Conv2D(32, (3, 3), strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6.0)(x)

    x = inverted_residual(x, expansion=1, filters=16, stride=1)
    x = inverted_residual(x, expansion=6, filters=24, stride=2)
    x = inverted_residual(x, expansion=6, filters=32, stride=2)
    x = inverted_residual(x, expansion=6, filters=64, stride=2)
    x = inverted_residual(x, expansion=6, filters=96, stride=1)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="MobileNetV2_Scratch")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model
