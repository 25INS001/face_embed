FROM python:3.10-slim

# Environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps
# libgl1 and libglib2.0-0 are gone with opencv-python-headless: they existed
# only for the GUI build of opencv, which a server container never uses.
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python deps
COPY requirements.txt .
# --timeout/--retries because this pulls the torch stack (hundreds of MB) and
# the arm64 builder is on a slower link: the default 15s socket timeout was
# enough to fail the build outright with a ReadTimeoutError from PyPI.
# --extra-index-url pins torch/torchvision to CPU wheels. Without it, pip
# resolves the default CUDA builds and drags the nvidia-* stack back in --
# 4.9GB of libraries the pods have no GPU to run.
RUN pip install --no-cache-dir --timeout 120 --retries 10 \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Copy project
COPY . .

# Create model cache dir
RUN mkdir -p /app/models

EXPOSE 8000

# Run with gunicorn
CMD ["gunicorn", "face_embed.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "2", "--threads", "4"]
