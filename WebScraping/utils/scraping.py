import requests
from tqdm import tqdm
from bs4 import BeautifulSoup

def getSoup(url):
    r = requests.get(url)
    r.encoding = r.apparent_encoding
    soup = BeautifulSoup(r.text,"html.parser").find_all('ul')

    return soup

def getURLlist(soup):
    url_list=[]
    for tmp_soup in soup:
        for new_a in tmp_soup.find_all('a',class_='new'):
            new_a.decompose()
        for tmp in tmp_soup.find_all('a'):
            url_list.append(tmp.get('href'))
    return url_list

#説明文をそれぞれ抽出
def extractSentence(url_list):
    text_lists=[]
    title_lists=[]
    for url in tqdm(url_list):
        url='https://ja.wikipedia.org'+str(url)
        tmp_r = requests.get(url)
        tmp_r.encoding = tmp_r.apparent_encoding

        text_lists.append(BeautifulSoup(tmp_r.text,"html.parser").find_all('p'))
        title_lists.append(BeautifulSoup(tmp_r.text,"html.parser").find('h1').get_text())
    
    return text_lists,title_lists

#それぞれの人物で文章を結合
def concatSentence(text_lists):
    exp_list=[]
    for text_list in tqdm(text_lists):
        tmp_text=''
        for text in text_list:
            tmp_text=tmp_text+text.get_text()
        exp_list.append(tmp_text)
    
    return exp_list