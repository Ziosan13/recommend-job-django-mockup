import requests
import pandas as pd
from bs4 import BeautifulSoup
from utils.scraping import getSoup, getURLlist, extractSentence, concatSentence
from utils.preprocess import Preprocess_df

#wikipediaの歴史上の人物から探索
url='https://ja.wikipedia.org/wiki/%E6%AD%B4%E5%8F%B2%E4%B8%8A%E3%81%AE%E4%BA%BA%E7%89%A9%E4%B8%80%E8%A6%A7'
r = requests.get(url)
r.encoding = r.apparent_encoding
soup = BeautifulSoup(r.text,"html.parser").find_all('ul')

#日本の歴史上人物が並べられた親urlを取得
url_parent=soup[22].find_all('a')

category_num=[1,2,3,8,44,31]
category_list=['fujiwara','genji','heisi','kajin','syosetu','meiji']
urlrange_list=[[18,22],[24,61],[24,31],[24,34],[13,21],[25,35]]

#カテゴリごとにスクレイピング
for i, urlrange in enumerate(urlrange_list):
    category='https://ja.wikipedia.org'+url_parent[category_num[i]].get('href')
    soup_category=getSoup(category)
    urllist_category=getURLlist(soup_category[urlrange[0]:urlrange[1]]) #getURLlistの引数は、該当する部分だけを自分で調べる
    
    text_category, title_category=extractSentence(urllist_category) #元データから名前と説明文を抽出
    exp_category=concatSentence(text_category) #各人物に対して一つの文章になるようにリスト内の文章を全結合

    df_category=pd.DataFrame(data={
        'name':title_category,
        'text':exp_category,
        'label': [i for j in range(len(title_category))]
        })

    df_category=df_category.apply(Preprocess_df,axis=1)
    df_category.to_csv('/workspace/WebScraping/csv/'+category_list[i]+'.csv')