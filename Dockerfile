FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

ENV FLASK_APP=main.py
ENV FLASK_RUN_HOST=0.0.0.0

EXPOSE 500

CMD ["flask", "run"]