# Quiz/02.py 코드를 tensorflow 사용하지 않고 수기로 작성한 코드
import pandas as pd

data = pd.read_csv('data/titles.csv')
titles = data['title'].values

print(titles)

word_index = dict()
for title in titles:
    words = title.split()
    for word in words:
        if word not in word_index:
            word_index[word] = len(word_index) + 1

print(word_index)
print(len(word_index))