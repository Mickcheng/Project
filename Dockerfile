FROM python:3.6

WORKDIR C:/
RUN mkdir ./dataprojects
RUN mkdir ./dataprojects/model

ENV MODEL_DIR=/C:/dataprojects/model/
ENV MODEL_FILE=SVM.joblib
ENV METADATA_FILE =metadata.json

ENV FLASK_APP=app.py

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .


COPY docker-ml.py .
RUN python docker-ml.py


EXPOSE 5000

CMD ["python", "app.py"]