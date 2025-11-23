FROM python:3.11-slim
WORKDIR /app

RUN pip install kubernetes
COPY scheduler.py .

CMD ["python","-u", "scheduler.py"]
