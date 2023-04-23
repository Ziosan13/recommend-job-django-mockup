#ライブラリ
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

#文字制限は512文字以内です。

def AI(sentence):
    
    #モデルの読み込み
    save_directory='./app/pretrained'
    model = BertForSequenceClassification.from_pretrained(save_directory, num_labels=6)
    tokenizer = BertTokenizer.from_pretrained(save_directory)
    
    sentiment_analyzer = pipeline("text-classification", model=model.to('cpu'), tokenizer=tokenizer)
    label=int(sentiment_analyzer(sentence)[0]['label'][-1])
    
    if label==0:
        answer='あなたは藤原氏タイプです！'
    elif label==1:
        answer='あなたは源氏タイプです！'
    elif label==2:
        answer='あなたは平氏タイプです！'
    elif label==3:
        answer='あなたは歌人タイプです！'
    elif label==4:
        answer='あなたは小説家タイプです！'
    else:
        answer='あなたは明治時代タイプです！'
        
    return answer