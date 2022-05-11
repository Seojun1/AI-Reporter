import tensorflow as tf
import pandas as pd

list1 = []
data = pd.read_csv('../data/titles.csv')
titles = data['title'].values

# preprocessing -> 전처리 단계 (학습하기 전에 처리하는 단계) / 데이터 가공 과정
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(titles)

x = []
y = []
for i in range(len(titles)):
    # sequence -> 수열 / 어떤 수의 나열로 바꿔서 표현해주는 것
    sequence = tokenizer.texts_to_sequences([titles[i]])[0]
    for j in range(1, len(sequence)):
        x.append(sequence[:j])
        y.append(sequence[j])

print(x) # x의 값
print(y) # y의 값

