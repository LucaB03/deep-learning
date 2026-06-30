import tensorflow as tf
from keras import layers, models

# Gemeinsame Data Augmentation für alle Modelle (gegen Overfitting)
def get_augmentation_layer():
    return models.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomContrast(0.1)
    ], name="data_augmentation")