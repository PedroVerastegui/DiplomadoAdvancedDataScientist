FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Genera datos y entrena el modelo al construir la imagen (para una demo autocontenida).
RUN python main.py make-data && python main.py train

EXPOSE 8000
CMD ["uvicorn", "src.service:app", "--host", "0.0.0.0", "--port", "8000"]
