# Imagen de la API. Se construye desde la RAÍZ del repositorio porque la API
# necesita src/nlp/preprocesamiento.py y los artefactos del modelo:
#   docker build -f api/Dockerfile -t hoteles-api .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    PORT=8000

WORKDIR /app

COPY api/requirements-docker.txt .
RUN pip install -r requirements-docker.txt

# joblib guarda una referencia a src.nlp.preprocesamiento.TextoNormalizador: debe ser importable.
COPY src/__init__.py src/
COPY src/nlp/__init__.py src/nlp/preprocesamiento.py src/nlp/
COPY artefactos/modelo_final.joblib artefactos/metricas_modelado.json artefactos/
COPY api/__init__.py api/main.py api/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# PORT lo definen plataformas como Cloud Run o Render; en local vale 8000.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
