FROM python:3.11

WORKDIR /app

COPY requirements.txt ./
COPY app.py ./

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8501

ENTRYPOINT [ "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0" ]