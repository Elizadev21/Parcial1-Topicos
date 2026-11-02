"""
API del clasificador de sentimiento de reseñas de hoteles.

Carga UNA vez el pipeline completo (preprocesamiento + modelo) que guarda
src/models/entrenar.py en artefactos/modelo_final.joblib, y lo usa con el
texto crudo que envía el usuario. No repite la lógica de NLP: los pasos que
muestra /api/preprocesar salen de src/nlp/preprocesamiento.pasos, la misma
función que usa el pipeline al entrenar y al predecir.

Documentación interactiva de los endpoints: /docs (Swagger) y /redoc.

Ejecutar desde la RAÍZ del repositorio (para poder importar src.nlp):

    uvicorn api.main:app --port 8000
"""
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.docs import get_redoc_html
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StrictStr, field_validator

from src.nlp.preprocesamiento import normalizar, pasos

MAX_CARACTERES = 5000

RAIZ = Path(__file__).resolve().parents[1]       # api -> raíz
MODELO = RAIZ / "artefactos" / "modelo_final.joblib"
METRICAS = RAIZ / "artefactos" / "metricas_modelado.json"

logger = logging.getLogger(__name__)
_modelo = None                                   # None = modelo no disponible


def cargar_modelo(ruta: Path = MODELO) -> None:
    """Carga el pipeline; si falla, la API sigue viva y /salud responde 503."""
    global _modelo
    try:
        _modelo = joblib.load(ruta)
    except Exception:
        logger.exception("No se pudo cargar el modelo desde %s", ruta)
        _modelo = None


def modelo_o_503():
    if _modelo is None:
        raise HTTPException(status_code=503, detail="El modelo no está disponible.")
    return _modelo


@asynccontextmanager
async def lifespan(app: FastAPI):
    cargar_modelo()
    yield


app = FastAPI(title="Sentimiento de reseñas de hoteles", version="1.0.0",
              description="Clasifica una reseña en negativo, neutro o positivo.",
              lifespan=lifespan, redoc_url=None)

# FastAPI carga ReDoc desde el CDN con la etiqueta "redoc@next", que ahora apunta a la
# versión 3 en pruebas (un módulo ES) y deja /redoc en blanco. Se fija la versión 2 estable.
REDOC_JS = "https://cdn.jsdelivr.net/npm/redoc@2.5.4/bundles/redoc.standalone.js"


@app.get("/redoc", include_in_schema=False)
def redoc():
    return get_redoc_html(openapi_url=app.openapi_url, title=f"{app.title} - ReDoc", redoc_js_url=REDOC_JS)


def _mensaje_validacion(error: dict) -> str:
    """Traduce un error de validación de pydantic a un mensaje claro en español."""
    tipo = error.get("type", "")
    if tipo == "missing" or tipo == "json_invalid" or tipo.endswith("_type"):
        return "Envía un JSON con el campo \"texto\" como cadena de texto, por ejemplo {\"texto\": \"Muy buen hotel\"}."
    if tipo == "string_too_short":
        return "La reseña está vacía."
    if tipo == "string_too_long":
        return f"La reseña supera los {MAX_CARACTERES} caracteres."
    if tipo == "value_error":
        return str(error.get("ctx", {}).get("error", "Texto no válido."))
    return "La solicitud no es válida."


@app.exception_handler(RequestValidationError)
async def error_de_validacion(request: Request, exc: RequestValidationError):
    errores = exc.errors()
    detalle = _mensaje_validacion(errores[0]) if errores else "La solicitud no es válida."
    return JSONResponse(status_code=422, content={"detail": detalle})


class Resena(BaseModel):
    texto: StrictStr = Field(min_length=1, max_length=MAX_CARACTERES,
                             examples=["La habitación era amplia y el personal muy atento."])

    @field_validator("texto")
    @classmethod
    def texto_legible(cls, valor: str) -> str:
        # Mismo criterio que el pipeline: si tras normalizar no queda nada, no hay qué clasificar.
        if not normalizar(valor):
            raise ValueError("Escribe una reseña con palabras.")
        return valor.strip()


@app.get("/api/salud", summary="Estado del servicio y del modelo",
         responses={503: {"description": "El modelo no está disponible"}})
def salud():
    """Devuelve 200 si la API y el modelo funcionan; 503 si el modelo no se pudo cargar."""
    modelo_o_503()
    return {"estado": "ok", "modelo": MODELO.name}


@app.post("/api/predecir", summary="Predecir el sentimiento de una reseña",
          responses={422: {"description": "Texto vacío, demasiado largo o con formato incorrecto"},
                     503: {"description": "El modelo no está disponible"}})
def predecir(resena: Resena):
    """Devuelve la clase (negativo, neutro o positivo), su probabilidad, las
    probabilidades de cada clase y los pasos del preprocesamiento. Si ninguna
    palabra está en el vocabulario del modelo, no clasifica y devuelve un aviso."""
    modelo = modelo_o_503()
    preprocesamiento = pasos(resena.texto)
    # Mismo pipeline que modelo.predict_proba, separado en sus dos pasos para
    # detectar textos sin ninguna palabra del vocabulario (p. ej. "jajaja"):
    # en ese caso el modelo solo devolvería la proporción de clases del entrenamiento.
    rasgos = modelo.named_steps["prep"].transform([resena.texto])
    if rasgos.nnz == 0:
        return {"sentimiento": None, "confianza": None, "probabilidades": {},
                "preprocesamiento": preprocesamiento,
                "aviso": "El modelo no reconoce ninguna palabra de la reseña; escribe más detalles."}
    clasificador = modelo.named_steps["clf"]
    probabilidades = clasificador.predict_proba(rasgos)[0]
    por_clase = {str(c): round(float(p), 4) for c, p in zip(clasificador.classes_, probabilidades)}
    sentimiento = max(por_clase, key=por_clase.get)
    return {
        "sentimiento": sentimiento,
        "confianza": por_clase[sentimiento],
        "probabilidades": por_clase,
        "preprocesamiento": preprocesamiento,
        "aviso": None,
    }


@app.post("/api/preprocesar", summary="Mostrar el preprocesamiento paso a paso",
          responses={422: {"description": "Texto vacío, demasiado largo o con formato incorrecto"}})
def preprocesar(resena: Resena):
    """Texto normalizado, tokens, stop words eliminadas y raíces (stemming): las
    mismas etapas que aplica el modelo antes del TF-IDF. No necesita el modelo."""
    return pasos(resena.texto)


@app.get("/api/metricas", summary="Métricas del modelo y comparación de modelos",
         responses={503: {"description": "Métricas no disponibles"}})
def metricas():
    """Validación cruzada de los modelos comparados y evaluación única en el conjunto de prueba."""
    try:
        datos = json.loads(METRICAS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise HTTPException(status_code=503, detail="Métricas no disponibles.")
    return {clave: datos[clave] for clave in
            ("modelo_en_produccion", "mejores_parametros", "validacion_cruzada", "prueba", "clases")
            if clave in datos}
