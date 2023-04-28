# recommend-job-django-mockup

## 概要

東大ソフトバンクハッカソンでのモックアップ

内容として

- 障害者雇用に特化した人財派遣サービスの展開
- サービスのうち、自分のプロフィールを記入すると向いてる職種・業界を表示する機能のモックアップ
- Wikipediaで公開されている、日本の歴史上の人物を学習データとして、記入したプロフィールが歴史上の人物のどのタイプに近いのかを提示してくれる
- フロント機能の実装


## セットアップ
### コンテナの立ち上げ
```
cd recommend-job-django-mockup
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
python WebScraping.py #実行時間がかなり長いので注意
python IntegrateCSV.py
```

### BERTで機械学習
```
cd /workspace/ML
python train.py
```

### djangoから操作
```
cd /workspace/instant-django
python manage.py runserver 0.0.0.0:8000
``` 

