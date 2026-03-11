FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY main.py .
COPY src/ src/

#copy dataset
COPY data/ ./data/

# Model artifacts are mounted at runtime or pre-trained
# COPY model_artifacts/ model_artifacts/   # uncomment after training

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]