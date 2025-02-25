FROM python:3.11.11-slim

WORKDIR /app

COPY ./app/requirements.txt /app/

RUN pip install -r requirements.txt --default-timeout=900

# RUN apt-get update && apt-get install -y curl

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
