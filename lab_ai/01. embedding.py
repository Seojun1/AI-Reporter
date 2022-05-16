import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(3, 5)
])
model.summary()