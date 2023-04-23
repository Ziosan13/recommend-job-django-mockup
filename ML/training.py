#ライブラリのインポート
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

#データ読み込み
df=pd.read_csv('/workspace/ML/ijin.csv',index_col=0,encoding='utf-8')
def to_str(df):
    return str(df['text'])
df['text']=df.apply(to_str,axis=1)

df=df.reset_index()
df=df.drop(['index'],axis=1)
df.head()

#学習済みモデル読み込み
model_name = "cl-tohoku/bert-base-japanese"

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=6)
tokenizer = BertTokenizer.from_pretrained(model_name)

#学習用と評価用に分ける
df_train=df.sample(frac=0.8)
df_eval=df.drop(df_train.index)

train_docs = df_train["text"].tolist()
train_labels = df_train["label"].tolist()

eval_docs=df_eval["text"].tolist()
eval_labels = df_eval["label"].tolist()

train_encodings = tokenizer(train_docs, return_tensors='pt', padding=True, truncation=True, max_length=128)
eval_encodings = tokenizer(eval_docs, return_tensors='pt', padding=True, truncation=True, max_length=128)

class JpSentiDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = JpSentiDataset(train_encodings, train_labels)
eval_dataset = JpSentiDataset(eval_encodings, eval_labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

training_args = TrainingArguments(
    output_dir='./results',          # 結果を出力するディレクトリ
    num_train_epochs=1,              # 学習回数の変更、計算資源が潤沢にある場合は変更を推奨
    per_device_train_batch_size=4,   # トレーニング中のバッチサイズ
    per_device_eval_batch_size=4,    # 評価中のバッチサイズ
    warmup_steps=500,                # スケジューラーの学習率の初期値を設定するステップ数
    weight_decay=0.01,               
    save_total_limit=1,              
    dataloader_pin_memory=False,  
    evaluation_strategy="steps",
    logging_steps=50,
    logging_dir='./logs'
)

trainer = Trainer(
    model=model,                         # モデルの指定
    args=training_args,                  # パラメータの指定
    train_dataset=train_dataset,         # 学習用データセットの指定
    eval_dataset=eval_dataset,           # 評価用データセットの指定
    compute_metrics=compute_metrics      # 評価に使う関数の指定
)

trainer.train()



#モデル保存
save_directory='/workspace/instant-django/app/pretrained'
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)