FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY starter_kit/ ./starter_kit/

WORKDIR /app

EXPOSE 8080

CMD ["sh", "-c", "streamlit run starter_kit/app.py --server.address=0.0.0.0 --server.port=${PORT:-8080}"]