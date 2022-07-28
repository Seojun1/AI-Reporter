import tensorflow as tf
import pandas as pd
import numpy as np

list1 = []
max_len = 0

data = pd.read_csv('../data/titles.csv')
titles = data['title'].values

# preprocessing -> 전처리 단계 (학습하기 전에 처리하는 단계) / 데이터 가공 과정
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(titles)

x = []
y = []
for i in range(len(titles)):
    # sequence -> 수열 / 어떤 수의 나열로 바꿔서 표현해주는 것 (문장)
    sequence = tokenizer.texts_to_sequences([titles[i]])[0]
    e = np.array(sequence)
    n = len(sequence)
    for j in range(n-1):
        x.append(sequence[:j+1])
        y.append(e[j+1])

print(x) # x의 값
print(y) # y의 값
