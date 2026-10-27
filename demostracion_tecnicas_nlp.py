\
"""Genera ejemplos verificables de normalización, tokenización, stopwords,
stemming, lematización y TF-IDF usando las funciones reales del proyecto.
Ejecutar después de instalar dependencias y el modelo spaCy español.
"""
from pathlib import Path
import pandas as pd
import spacy
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocesamiento import normalizar, STOPWORDS_ES

BASE = Path(__file__).resolve().parent
OUT = BASE / "resultados_nlp"
OUT.mkdir(exist_ok=True)

EJEMPLOS = [
    "¡¡NO recomiendo este hotel!!! Visita https://ejemplo.com en 2024.",
    "La habitación estaba buenoooo, pero no estaba limpia.",
    "Nada limpio y nunca volvería.",
    "El servicio fue muy bueno, aunque la habitación no estaba limpia.",
]

# Cargar una sola vez el modelo español para mostrar lemas.
nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])
stemmer = SnowballStemmer("spanish")
filas = []
for original in EJEMPLOS:
    normalizado = normalizar(original)
    tokens = normalizado.split()
    tokens_sin_stopwords = [t for t in tokens if t not in STOPWORDS_ES]
    stemming = [stemmer.stem(t) for t in tokens_sin_stopwords]
    doc = nlp(" ".join(tokens_sin_stopwords))
    lemas = [tok.lemma_.lower() for tok in doc if tok.is_alpha]
    filas.append({
        "texto_original": original,
        "1_normalizacion": normalizado,
        "2_tokens": " | ".join(tokens),
        "3_tokens_sin_stopwords": " | ".join(tokens_sin_stopwords),
        "4_stemming": " | ".join(stemming),
        "5_lematizacion": " | ".join(lemas),
        "se_conserva_no": "no" in tokens_sin_stopwords,
    })

pd.DataFrame(filas).to_csv(OUT / "ejemplos_preprocesamiento.csv", index=False, encoding="utf-8-sig")

# Evidencia de TF-IDF con una muestra didáctica. Esto es solo demostración,
# no sustituye al ajuste TF-IDF sobre el conjunto train real.
corpus = [normalizar(x) for x in EJEMPLOS]
vectorizador = TfidfVectorizer(stop_words=sorted(STOPWORDS_ES), ngram_range=(1, 2), min_df=1, sublinear_tf=True)
matriz = vectorizador.fit_transform(corpus)
terminos = vectorizador.get_feature_names_out()
# IDF es un peso del vocabulario; los valores de la matriz son TF-IDF por texto.
pd.DataFrame({"termino": terminos, "idf": vectorizador.idf_}).sort_values("idf", ascending=False).to_csv(
    OUT / "detalle_tfidf.csv", index=False, encoding="utf-8-sig")

print("Demostraciones guardadas en:", OUT)
print("- ejemplos_preprocesamiento.csv: pasos de NLP, incluyendo la conservación de 'no'.")
print("- detalle_tfidf.csv: términos e IDF de una muestra didáctica.")
print("Nota: detalle_tfidf.csv usa cuatro ejemplos ilustrativos; la comparación de modelos usa los datos train/test del proyecto.")
