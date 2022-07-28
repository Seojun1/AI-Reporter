import tensorflow as tf
import pandas as pd
import numpy as np

data = pd.read_csv('../data/titles.csv')
titles = data['title'].values

# tokenizer --> 뉴스 제목 문장을 구성하는 단어에 정수 인덱스 부여하는 속성
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(titles)

# word_count --> 단어의 개수 ( index는 0부터 시작하기 때문에 +1을 해줌 )
word_count = len(tokenizer.word_index) + 1

index_word = ['' for _ in range(word_count)]
for k, v in tokenizer.word_index.items():   # v = 1 / k = 이은해
    index_word[v] = k

# 여기까지가 도전과제 2번에서 text를 정수 인덱스 번호로 바꿨던 작업을 다시 text로 바꿔서 리스트에 append 하는 과정이다.

# 다음 말로 무엇이 들어와야되는지에 대한 예측값을 저장했던 모델 불러오기
model = tf.keras.models.load_model('../models/test.h5')

title = '이은해'
length = 10

for i in range(length):
    sequence = tokenizer.texts_to_sequences([title])[0]
    predict = model.predict([sequence])
    index = np.argmax(predict[0])
    title += ' ' + index_word[index]

print(title)