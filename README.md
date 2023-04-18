# recommend-job-django-mockup

## 概要


## セットアップ
### コンテナの立ち上げ
このリポジトリの階層に移動した後
```
docker build . -t recommend-job
```


コンテナイメージが立ち上がった後
```
cd WebScraping/csv
docker run -v `pwd`:/workspace/WebScraping/csv --name mockup -itd recommend-job
```
### djangoの設定
superuserの追加
コンテナに入ってから
```
cd /workspace/instant-django
python manage.py createsuperuser
```
## 各ステップの説明
### スクレイピング
```
cd /workspace/WebScraping
python WebScraping.py #実行時間かなり長め
python IntegrateCSV.py
```

### BERTで機械学習
```
cd /workspace/ML
python training.py
```

### フロントエンドに適用
```
cd /workspace/instant-django
python manage.py runserver
``` 

