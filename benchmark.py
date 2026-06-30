import os, time
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import keras

BATCH, IMG = 128, (256, 256, 3)

# Synthetic dataset: 10 batches in memory, no disk I/O
x = tf.random.uniform((10 * BATCH, *IMG))
y = tf.random.uniform((10 * BATCH,), 0, 45, dtype=tf.int32)
ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(BATCH).prefetch(tf.data.AUTOTUNE)

from models.leNet_5 import generateModel
augmenter = keras.Sequential([
    keras.layers.RandomFlip("horizontal_and_vertical"),
    keras.layers.RandomRotation(0.5),
    keras.layers.RandomZoom((-0.1, 0.1), (-0.1, 0.1)),
    keras.layers.RandomTranslation(0.1, 0.1),
    keras.layers.RandomContrast(0.15),
])
model = generateModel(IMG)
wrapped = tf.keras.Sequential([augmenter, model])
wrapped.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

@tf.function
def step(x, y):
    with tf.GradientTape() as t:
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, wrapped(x, training=True)))
    wrapped.optimizer.apply_gradients(zip(t.gradient(loss, wrapped.trainable_variables), wrapped.trainable_variables))
    return loss

# warmup
for xb, yb in ds.take(2): step(xb, yb)

times = []
for xb, yb in ds.take(5):
    t0 = time.perf_counter()
    step(xb, yb)
    times.append((time.perf_counter() - t0) * 1000)

print(f"GPU step (aug in model): {np.mean(times):.1f} ms  min={min(times):.1f} max={max(times):.1f}")
