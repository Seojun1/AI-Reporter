import tensorflow as tf

model = tf.keras.models.load_model('../models/sum_softmax.h5')
model.summary()
