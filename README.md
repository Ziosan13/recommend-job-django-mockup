# recommend-job-django-mockup

# Dockerでのセットアップ
このリポジトリの階層に移動した後
```
docker build . -t recommend-job
```


コンテナイメージが立ち上がった後
```
cd WebScraping/csv
docker run -v `pwd`:/workspace/WebScraping/csv --name mockup -itd recommend-job
```