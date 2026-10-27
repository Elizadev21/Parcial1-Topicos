"""
preprocesamiento.py
===================
Pipeline de NLP del proyecto: "Normalizar a TF-IDF".

Este MISMO módulo se usa en dos momentos:
  1. Entrenamiento (preparar_datos.py y el trabajo de Modelado).
  2. Uso (el backend carga artefactos/preprocesador.joblib y lo aplica a cada
     texto que escribe el usuario).

Por eso el backend debe poder hacer `import preprocesamiento` (este archivo
tiene que estar en el mismo proyecto / en el PYTHONPATH del backend), porque
joblib guarda una referencia a la clase TextoNormalizador definida aquí.

Decisiones de diseño (justificadas en el documento de cambios):
  - Se quitan acentos (pero se conserva la ñ) porque en las reseñas hay
    textos con y sin tildes ("habitación" / "habitacion") y deben contar
    como la misma palabra.
  - Se quitan URLs, números y signos de puntuación.
  - Se reducen alargamientos ("buenooo" -> "bueno").
  - NO se eliminan negaciones ni intensificadores ("no", "sin", "nada",
    "nunca", "muy", "poco"...): en sentimiento cambian el sentido de la frase.
  - TF-IDF con unigramas + bigramas para capturar "no recomiendo", "nada limpio".
"""
import re
import unicodedata

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Stopwords en español (ya sin acentos, porque se aplican DESPUÉS de normalizar).
# Solo palabras funcionales (artículos, preposiciones, pronombres...).
# Se dejan FUERA a propósito: no, ni, sin, nada, nunca, jamas, tampoco, nadie,
# ningun, muy, mas, menos, poco, mucho, bastante, demasiado, pero, aunque, sino,
# solo, hay. Todas ellas aportan información de sentimiento.
# Solo se incluyen palabras de 2+ letras (el tokenizador de sklearn ignora las
# de 1 letra, y así evitamos advertencias de inconsistencia).
# ---------------------------------------------------------------------------
STOPWORDS_ES = frozenset("""
al ante con contra de del desde durante en entre hacia hasta para por segun
sobre tras el la los las un una unos unas lo que se me te le les nos mi mis tu
tus su sus yo ella ellos ellas nosotros este esta estos estas ese esa esos esas
eso esto aquel aquella es son fue fueron era eran ser ha han he hemos habia
como cuando donde quien cual si ya tambien
""".split())

_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_NUMEROS = re.compile(r"\d+")
_RE_ALARGADOS = re.compile(r"(.)\1{2,}")          # "buenooo" -> "bueno"
_RE_NO_LETRAS = re.compile(r"[^a-zñ\s]")
_RE_ESPACIOS = re.compile(r"\s+")


def _quitar_acentos(texto: str) -> str:
    """Quita tildes y diéresis pero conserva la ñ."""
    nfd = unicodedata.normalize("NFD", texto)
    salida = []
    for ch in nfd:
        if unicodedata.category(ch) == "Mn":       # marca combinante (tilde, etc.)
            if ch == "\u0303" and salida and salida[-1] == "n":
                salida.append(ch)                  # es la tilde de la ñ: se conserva
            continue
        salida.append(ch)
    return unicodedata.normalize("NFC", "".join(salida))


def normalizar(texto) -> str:
    """Normaliza UNA reseña: minúsculas, sin URLs/números/signos/acentos."""
    if not isinstance(texto, str):
        return ""
    t = texto.lower()
    t = _RE_URL.sub(" ", t)
    t = _quitar_acentos(t)
    t = _RE_NUMEROS.sub(" ", t)
    t = _RE_ALARGADOS.sub(r"\1", t)
    t = _RE_NO_LETRAS.sub(" ", t)
    return _RE_ESPACIOS.sub(" ", t).strip()


class TextoNormalizador(BaseEstimator, TransformerMixin):
    """Paso de sklearn que aplica `normalizar` a una colección de textos."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, str):                     # un solo texto (caso del backend)
            X = [X]
        return [normalizar(t) for t in X]


def crear_preprocesador() -> Pipeline:
    """Devuelve el pipeline SIN entrenar: normalización -> TF-IDF."""
    return Pipeline([
        ("normalizar", TextoNormalizador()),
        ("tfidf", TfidfVectorizer(
            stop_words=sorted(STOPWORDS_ES),
            ngram_range=(1, 2),   # palabras sueltas y pares ("no recomiendo")
            min_df=2,             # ignora términos que aparecen en 1 sola reseña
            sublinear_tf=True,    # atenúa palabras muy repetidas en una reseña
        )),
    ])
