FROM python:3.12.3

RUN apt update -y && apt install -y curl

WORKDIR /dependences
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app
COPY ./src .

WORKDIR /data
COPY ./docs .

WORKDIR /app

CMD ["python3", "-u", "main.py"]
