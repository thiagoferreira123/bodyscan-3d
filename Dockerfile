FROM python:3.11-slim AS base

WORKDIR /app

# System deps for image processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps (torch CPU-only from PyTorch index)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Download ML models from BunnyCDN
RUN mkdir -p /app/models && \
    curl -fSL -o /app/models/stage_a_v2_best.pth \
      "https://ds-course.b-cdn.net/stage_a_v2_best.pth" && \
    curl -fSL -o /app/models/stage_b__male.pkl \
      "https://ds-course.b-cdn.net/stage_b__male.pkl" && \
    curl -fSL -o /app/models/stage_b__female.pkl \
      "https://ds-course.b-cdn.net/stage_b__female.pkl"

# Copy application code
COPY app/ /app/app/

ENV MODELS_DIR=/app/models
ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
