import tensorflow as tf
import numpy as np
import os

index_to_word = ['가', '나', '다', '라']

# 가나다라, 다나가가
x = [[0, 1, 2], [2, 1, 0]]
y = [[0, 0, 0, 1], [1, 0, 0, 0]]

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(4, 5),
    tf.keras.layers.SimpleRNN(3),
    tf.keras.layers.Dense(4),
    tf.keras.layers.Softmax()
])

# 손실함수 (학습이 잘 됬는지 안됬는지 알기 위함 --> 값이 낮으면 학습이 잘된 것으로 판단 가능)
loss = tf.keras.losses.CategoricalCrossentropy()
# 최적화 알고리즘 (Adam 사용 / 경사하강법을 사용하기 위함)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# accuracy = 정확도 / tensorboad는 정확도를 그래프로 볼 수 있기 때문에 정확도 변화과정을 이해하기 쉽게 볼 수 있음.
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

if not os.path.exists('../logs'):
    os.mkdir('../logs')

# tensorboard는 keras속 callbacks 안에 들어있음.
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='../logs')
model.fit(x, y, epochs=100, callbacks=[tensorboard])

if not os.path.exists('../models'):
    os.mkdir('../models')

model.save('../models/test.h5')