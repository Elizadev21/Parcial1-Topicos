"""
src/nlp/preprocesamiento.py
===========================
Pipeline de NLP del proyecto: "Normalizar a TF-IDF".

Este MISMO módulo se usa en dos momentos:
  1. Entrenamiento (src/nlp/preparar_datos.py y src/models/entrenar.py).
  2. Uso (la API carga artefactos/modelo_final.joblib y lo aplica a cada
     texto que escribe el usuario; /api/preprocesar muestra los pasos con
     la función `pasos`, que es la misma que usa el pipeline).

Por eso la API debe poder hacer `import src.nlp.preprocesamiento` (la raíz
del proyecto tiene que estar en el PYTHONPATH), porque joblib guarda una
referencia a la clase TextoNormalizador definida aquí.

Pasos, en orden:
  1. Normalización: minúsculas, sin acentos (se conserva la ñ), sin URLs,
     números ni signos; alargamientos reducidos ("buenooo" -> "bueno").
     Se quitan acentos porque hay textos con y sin tildes ("habitación" /
     "habitacion") y deben contar como la misma palabra.
  2. Tokenización: separa palabras; se descartan las de una sola letra
     ("y", "a", "o"), como hacía el TF-IDF por defecto.
  3. Stop words: se quitan palabras funcionales, pero NO negaciones ni
     intensificadores ("no", "sin", "nada", "nunca", "muy", "poco"...),
     porque en sentimiento cambian el sentido de la frase.
  4. Stemming (Snowball, español): reduce cada palabra a su raíz
     ("recomiendo" -> "recomiend"). Elegido frente a la lematización y a no
     reducir por F1 macro en validación cruzada (resultados_nlp/).
  5. TF-IDF con unigramas + bigramas para capturar "no recomiend", "nad limpi".
"""
import re
import unicodedata

from nltk.stem.snowball import SnowballStemmer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Stopwords en español (ya sin acentos, porque se aplican DESPUÉS de normalizar).
# Solo palabras funcionales (artículos, preposiciones, pronombres...).
# Se dejan FUERA a propósito: no, ni, sin, nada, nunca, jamas, tampoco, nadie,
# ningun, muy, mas, menos, poco, mucho, bastante, demasiado, pero, aunque, sino,
# solo, hay. Todas ellas aportan información de sentimiento.
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
_STEMMER = SnowballStemmer("spanish")


def _quitar_acentos(texto: str) -> str:
    """Quita tildes y diéresis pero conserva la ñ."""
    nfd = unicodedata.normalize("NFD", texto)
    salida = []
    for ch in nfd:
        if unicodedata.category(ch) == "Mn":       # marca combinante (tilde, etc.)
            if ch == "̃" and salida and salida[-1] == "n":
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


def pasos(texto) -> dict:
    """Aplica el preprocesamiento a UNA reseña y devuelve cada etapa.

    TextoNormalizador usa esta misma función, así que lo que se muestra al
    usuario es exactamente lo que recibe el TF-IDF del modelo.
    """
    normalizado = normalizar(texto)
    tokens = [t for t in normalizado.split() if len(t) > 1]
    sin_stopwords = [t for t in tokens if t not in STOPWORDS_ES]
    raices = [_STEMMER.stem(t) for t in sin_stopwords]
    return {
        "normalizado": normalizado,
        "tokens": tokens,
        "stopwords_eliminadas": [t for t in tokens if t in STOPWORDS_ES],
        "tokens_sin_stopwords": sin_stopwords,
        "raices": raices,
        "texto_procesado": " ".join(raices),
    }


class TextoNormalizador(BaseEstimator, TransformerMixin):
    """Paso de sklearn: normaliza, tokeniza, quita stop words y aplica stemming."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, str):                     # un solo texto (caso del backend)
            X = [X]
        return [pasos(t)["texto_procesado"] for t in X]


def crear_preprocesador() -> Pipeline:
    """Devuelve el pipeline SIN entrenar: preprocesamiento -> TF-IDF."""
    return Pipeline([
        ("normalizar", TextoNormalizador()),
        ("tfidf", TfidfVectorizer(
            token_pattern=r"\S+",  # los tokens ya vienen separados y filtrados por `pasos`
            ngram_range=(1, 2),    # palabras sueltas y pares ("no recomiend")
            min_df=2,              # ignora términos que aparecen en 1 sola reseña
            sublinear_tf=True,     # atenúa palabras muy repetidas en una reseña
        )),
    ])
