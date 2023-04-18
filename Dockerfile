FROM python:3.9-bullseye

RUN apt update && \
    apt install -y sudo \
    git 

#pythonのライブラリインストール
ENV PIP_NO_CACHE_DIR=off
RUN mkdir /code
WORKDIR /code
COPY requirements.txt /code/ 
COPY django/ /code/django/
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

RUN mkdir /workspace
WORKDIR /workspace
COPY ML/ /workspace/ML/
COPY WebScraping/ /workspace/WebScraping/

RUN git clone https://github.com/okoppe8/instant-django.git && \
    cd instant-django && \
    pip install -r requirements.txt && \
    mv /code/django/ai.py /workspace/instant-django/app/ai.py && \
    mv /code/django/views.py /workspace/instant-django/app/views.py && \
    mv /code/django/models.py /workspace/instant-django/app/models.py && \
    mv /code/django/inference.py /workspace/instant-django/app/inference.py && \
    mv /code/django/templates/app/item_detail_contents.html /workspace/instant-django/app/templates/app/item_detail_contents.html && \
    mv /code/django/templates/app/item_filter.html /workspace/instant-django/app/templates/app/item_filter.html && \
    python manage.py makemigrations && \
    python manage.py migrate