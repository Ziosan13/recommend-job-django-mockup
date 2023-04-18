#ライブラリのインポート
import re
import unicodedata
import pandas as pd

tag_sub = re.compile(r"<[^>]*?>")
def Preprocess(text):
    text=unicodedata.normalize('NFKC', text.strip())
    text=re.sub(r'[【】]', ' ', text)       # 【】の除去
    text=re.sub(r'[「」]', ' ', text)
    text=re.sub(r'[（）()]', ' ', text)     # （）の除去
    text=re.sub(r'[［\］[\]]', ' ', text)
    text=re.sub(r'[A-z]+', "", text)
    text=text.replace('\u3000','')
    text=text.replace(' ', '')
    text=text.replace(',', '')
    text=text.replace('\n', '')
    text=text.replace('\r', '')
    text=tag_sub.sub("",text)
    return str(text)

def Preprocess_df(df):
    return Preprocess(df['text'])