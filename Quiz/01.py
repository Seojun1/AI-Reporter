from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import ssl
import os
import pandas as pd

context = ssl._create_unverified_context()
headers = {'User-Agent': 'Mozilla/5.0'}

url = 'https://news.naver.com/main/clusterArticles.naver?id=c_202204291640_00000070&mode=LSD&mid=shm&sid1=102&oid=011&aid=0004049188'
request = Request(url, headers=headers)
response = urlopen(request, context=context)
html = response.read()

soup = BeautifulSoup(html, 'html.parser')
result = soup.find_all('a', {'class': 'nclicks(cls_nav.clsart1)'}) # a 태그의 클래스가 nclicks(cls_nav.clsart1) 인 것만 검출
titles = []

for r in result:
    # 제목이 아닌 다른 요소를 걸러내기 위한 코드
    if (len(r.text) < 9):
        continue
    else:
        print(r.text)
        titles.append(r.text)

# data 파일에 csv파일로 크롤링 내용 추가
if not os.path.exists('../data'):
    os.mkdir('../data')

data = pd.DataFrame({'title': titles})
data.to_csv('../data/titles.csv', encoding='utf-8')
print(titles)