# CPU image for the API. Build: docker build -t currency-api .
# Pinned to Debian bookworm so the apt package names below stay valid.
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# OpenCV (pulled in by ultralytics) needs these system libraries
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch first: much smaller image than the default CUDA build
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY app/ app/
COPY src/ src/
COPY configs/ configs/
RUN mkdir -p weights

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
