import tensorflow as tf

model_Embedding = tf.keras.models.load_model('../models/Embedding.h5')
model_Dense = tf.keras.models.load_model('../models/Dense.h5')
model_SimpleRNN = tf.keras.models.load_model('../models/SimpleRNN.h5')
model_Softmax = tf.keras.models.load_model('../models/Softmax.h5')

# 출력
predict_Embedding = model_Embedding.predict([[0, 1, 2], [1, 1, 1]])
print('Embedding 예측값 - \n', predict_Embedding)

model_SimpleRNN = model_SimpleRNN.predict([[0, 1, 2], [0, 1, 2]])
print('SimpleRNN 예측값 - \n', model_SimpleRNN)

predict_Dense = model_Dense.predict([[0, 1, 2], [0, 1, 2]])
print('Dense 예측값 - \n', predict_Dense)

model_Softmax = model_Softmax.predict([[0, 1, 2], [0, 1, 2]])
print('Softmax 예측값 - \n', model_Softmax)