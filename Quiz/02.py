# tensorflow에서 제공해주는 기능으로 데이터 토큰화 할 예정
import tensorflow as tf
import pandas as pd

data = pd.read_csv('../data/titles.csv')
titles = data['title'].values

# 가공처리 과정
# tokenizer = tf.keras.preprocessing.text.Tokenizer()
# tokenizer.fit_on_texts(titles)
# word_count = len(tokenizer.word_index) + 1
# print(tokenizer.word_index)
# print(word_count)

word_index = dict()