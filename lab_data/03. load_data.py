# pandas --> 데이터를 관리하는 용도 / 데이터처리에 특화된 패키지 (데이터처리)
import pandas as pd

data = pd.read_csv('../data/titles.csv')

# print(data['title'])
titles = data['title'].values
print(titles)