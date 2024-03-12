FROM python:3.12-slim

WORKDIR /app

ENV PYTHONPATH /app/src


RUN apt-get update && apt-get install -y build-essential && \
  apt-get clean

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

ENTRYPOINT [ "sh", "-c" ]

CMD [ "uvicorn main:app --host=0.0.0.0 --log-config=/app/src/log_conf.yaml" ]
