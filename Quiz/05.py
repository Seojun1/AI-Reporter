import tensorflow as tf
import os

model1 = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(3, 5),
])

model2 = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(3, 5),
    tf.keras.layers.Input(shape=(None, 8)),
    tf.keras.layers.SimpleRNN(4),
])

model3 = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(3, 5),
    tf.keras.layers.Input(shape=(None, 8)),
    tf.keras.layers.SimpleRNN(4),
    tf.keras.layers.Dense(6),
])

model4 = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(3, 5),
    tf.keras.layers.Input(shape=(None, 8)),
    tf.keras.layers.SimpleRNN(4),
    tf.keras.layers.Dense(6),
    tf.keras.layers.Softmax(),
])

if not os.path.exists('../models'):
    os.mkdir('../models')

model1.save('../models/Embedding.h5')
model2.save('../models/SimpleRNN.h5')
model3.save('../models/Dense.h5')
model4.save('../models/Softmax.h5')
