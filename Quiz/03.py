import tensorflow as tf
import pandas as pd
import numpy as np

list1 = []
data = pd.read_csv('../data/titles.csv')
titles = data['title'].values

# preprocessing -> 전처리 단계 (학습하기 전에 처리하는 단계) / 데이터 가공 과정
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(titles)
# print(tokenizer.word_index)

x = [] # 최종 x 배열 생성 (2차원 배열)
xx = [] # 2차원 배열에 append하기 위한 중간다리 배열

# sequence -> 수열 / 어떤 수의 나열로 바꿔서 표현해주는 것
sequence = tokenizer.texts_to_sequences(titles)
# for l in range(len(sequence)):
#         q = sequence[:l]
#         print(q)
        # for i in range(len(q)):
        #         xx.append(sequence[i])
        #         x.append(xx.copy())
q = sequence[:1]
print(q)
# print(x)
# print(sequence[1:]) # y값