FROM python:3.6

WORKDIR C:/
RUN mkdir ./dataproject
RUN mkdir ./dataproject/model

ENV MODEL_DIR=/C:/dataproject/model/
ENV MODEL_FILE=lr.joblib
ENV METADATA_FILE =metadata.json

ENV FLASK_APP=app.py

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .


COPY docker-ml.py .
RUN python docker-ml.py


EXPOSE 5000

CMD ["python", "app.py"]