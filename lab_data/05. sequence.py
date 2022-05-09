import tensorflow as tf
import pandas as pd

data = pd.read_csv('../data/titles.csv')
titles = data['title'].values

# preprocessing -> 전처리 단계 (학습하기 전에 처리하는 단계) / 데이터 가공 과정
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(titles)
# print(tokenizer.word_index)

# sequence -> 수열 / 어떤 수의 나열로 바꿔서 표현해주는 것
sequence = tokenizer.texts_to_sequences([titles[0]])[0]
print(titles[0])
print(sequence)