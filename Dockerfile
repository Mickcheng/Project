FROM python:3.6

WORKDIR C:/

RUN mkdir ./dataengineering
RUN mkdir ./dataengineering/model

ENV MODEL_DIR=/C:/dataengineering/model
ENV MODEL_FILE=lr.joblib
ENV METADATA_FILE =metadata.json


COPY requirements.txt .

RUN pip install -r requirements.txt

COPY Tweets.csv .

COPY docker-ml.py .
COPY docker-ml-inference.py .

RUN python docker-ml.py
RUN python docker-ml-inference.py