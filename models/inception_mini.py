import keras
import tensorflow as tf
from keras import Sequential, layers, models
from models.lib import get_augmentation_layer

def generate_model(input_shape=(256, 256, 3), num_classes=45):
    def inception_block(x, f_1x1, f_3x3_r, f_3x3, f_5x5_r, f_5x5, f_pool):
        p1 = layers.Conv2D(f_1x1, (1, 1), padding="same", activation="relu")(x)
        p2 = layers.Conv2D(f_3x3_r, (1, 1), padding="same", activation="relu")(x)
        p2 = layers.Conv2D(f_3x3, (3, 3), padding="same", activation="relu")(p2)
        p3 = layers.Conv2D(f_5x5_r, (1, 1), padding="same", activation="relu")(x)
        p3 = layers.Conv2D(f_5x5, (5, 5), padding="same", activation="relu")(p3)
        p4 = layers.MaxPooling2D((3, 3), strides=1, padding="same")(x)
        p4 = layers.Conv2D(f_pool, (1, 1), padding="same", activation="relu")(p4)
        return layers.concatenate([p1, p2, p3, p4], axis=-1)

    inputs = layers.Input(shape=input_shape)
    x = get_augmentation_layer()(inputs)

    x = layers.Conv2D(64, (7, 7), strides=2, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding="same")(x)

    x = inception_block(x, f_1x1=32, f_3x3_r=32, f_3x3=64, f_5x5_r=8, f_5x5=16, f_pool=16)
    x = layers.MaxPooling2D((3, 3), strides=2, padding="same")(x)
    x = inception_block(x, f_1x1=64, f_3x3_r=64, f_3x3=128, f_5x5_r=16, f_5x5=32, f_pool=32)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="InceptionMini_Scratch")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
