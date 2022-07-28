import tensorflow as tf
import pandas as pd
import numpy as np
import os

data = pd.read_csv('../data/titles.csv')
titles = data['title'].values

# 가공처리 과정
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(titles)
word_count = len(tokenizer.word_index) + 1

sequences = []
max_len = 0

x = []
y = []
for i in range(len(titles)):
    # sequence -> 수열 / 어떤 수의 나열로 바꿔서 표현해주는 것 (문장)
    sequence = tokenizer.texts_to_sequences([titles[i]])[0]
    e = np.array(sequence)
    n = len(sequence)
    max_len = max(max_len, len(sequence))
    for j in range(n-1):
        x.append(sequence[:j+1])
        y.append(e[j+1])

pad_sequences = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=max_len)
categorical_data = tf.keras.utils.to_categorical(y, num_classes=word_count)

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(word_count, 10),
    tf.keras.layers.SimpleRNN(50),
    tf.keras.layers.Dense(word_count),
    tf.keras.layers.Softmax(),
])
model.summary()

# 손실함수 (학습이 잘 됬는지 안됬는지 알기 위함 --> 값이 낮으면 학습이 잘된 것으로 판단 가능)
loss = tf.keras.losses.CategoricalCrossentropy()
# 최적화 알고리즘 (Adam 사용 / 경사하강법을 사용하기 위함)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# accuracy = 정확도 / tensorboad는 정확도를 그래프로 볼 수 있기 때문에 정확도 변화과정을 이해하기 쉽게 볼 수 있음.
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
# 학습을 하기 위한 훈련 데이터 삽입
model.fit(pad_sequences, categorical_data, epochs=100)

if not os.path.exists('../models'):
    os.mkdir('../models')

model.save('../models/test.h5')
