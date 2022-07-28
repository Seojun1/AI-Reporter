import tensorflow as tf
import pandas as pd
import numpy as np

data = pd.read_csv('../data/titles.csv')
titles = data['title'].values

# 가공처리 과정
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(titles)
word_count = len(tokenizer.word_index) + 1

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(word_count, 10),
    tf.keras.layers.SimpleRNN(32),
    tf.keras.layers.Dense(word_count),
    tf.keras.layers.Softmax(),
])
model.summary()

predict_model = model.predict([[0, 1, 2]])
print(predict_model)

index = np.argmax(predict_model[0]) # 예측 값
print(index)
