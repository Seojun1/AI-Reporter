# 전체 문장만을 위한 도전문제3번 ( 전체 문장이 전부 해당되기 때문에 효율성이 좋다 ) - 미완성

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

# sequence -> 수열 / 어떤 수의 나열로 바꿔서 표현해주는 것
sequence = tokenizer.texts_to_sequences(titles)
print(sequence[:1])

# x = []
# xx = []
#
# for i in range(len(sequence)):
#     b = np.array(
#         [
#             x.append(sequence[i])
#         ]
#     )
#     # 배열 자료형 --> 배열을 복사할때 append를 사용하면 참조복사(최종 결괏값만 복사)를 하기 때문에 발생한 문제를 copy로 사용하면 해결되는 문제였다.
#     xx.append(x.copy())
# print('x값 : ', xx)
#
# # y 값
# y = sequence[1:]
# print('y값 : ', y)