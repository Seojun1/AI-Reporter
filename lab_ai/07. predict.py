import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('../models/Softmax.h5')

predict = model.predict([[0, 1, 2], [1, 1, 1]])
print(predict)

index_word = ['가', '나', '다', '라', '마', '바']

index = np.argmax(predict[0])

print(index)
print(index_word[index])