FROM python:3.10-slim

# Environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create model cache dir
RUN mkdir -p /app/models

EXPOSE 8000

# Run with gunicorn
CMD ["gunicorn", "face_embed.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "2", "--threads", "4"]
