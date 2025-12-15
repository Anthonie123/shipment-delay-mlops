# Image dasar Python yang ringan
FROM python:3.11-slim

# Set working directory di dalam container
WORKDIR /app

# Copy requirements dan install dependency dulu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua kode project ke dalam container
COPY . .

# Expose port untuk FastAPI
EXPOSE 8000

# Jalankan FastAPI dengan Uvicorn
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
