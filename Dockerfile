FROM python:3.12.8-slim
WORKDIR /app
COPY . /app

RUN app update -y && apt install awscli -y

RUN pip install -r requirements.txt
CMD ["python3","app.py"]
