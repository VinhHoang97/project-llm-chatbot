FROM python:latest

RUN mkdir /api

COPY requirements.txt /api

WORKDIR /api

RUN pip install -r requirements.txt

COPY . /api

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]