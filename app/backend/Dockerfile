# Build with the REPO ROOT as context (docker compose does this), so the genai package installs:
#   docker build -f app/backend/Dockerfile .
FROM python:3.12-slim
WORKDIR /srv

# CPU-only torch (small wheel; genai.pets.corruptions needs torch tensors, no GPU needed).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY app/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The shared package, so runtime corruptions are the exact training code.
COPY pyproject.toml ./genai_pkg/pyproject.toml
COPY src ./genai_pkg/src
RUN pip install --no-cache-dir ./genai_pkg

COPY app/backend/app ./app
COPY app/backend/samples ./samples
ENV MODELS_DIR=/models SAMPLES_DIR=/srv/samples
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
