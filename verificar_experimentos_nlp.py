"""Verifica resultados y que las seis configuraciones estén completas."""
from pathlib import Path
import sys
import pandas as pd
BASE = Path(__file__).resolve().parent
OUT = BASE / "resultados_nlp"
required_files = [OUT/"comparacion_modelos_nlp.csv", OUT/"ejemplos_preprocesamiento.csv", OUT/"informe_comparaciones.md", OUT/"detalle_tfidf.csv"]
missing = [p.name for p in required_files if not p.exists()]
if missing:
    print("FALTAN ARCHIVOS:", ", ".join(missing))
    print("Ejecuta primero: python experimentos_nlp.py y python demostracion_tecnicas_nlp.py")
    sys.exit(1)
df = pd.read_csv(OUT/"comparacion_modelos_nlp.csv")
required = {(m,n) for m in ("ninguno","stemming","lematizacion") for n in ("unigramas","unigramas_bigramas")}
found = set(zip(df["metodo"], df["representacion"]))
if not required.issubset(found):
    print("FALTAN CONFIGURACIONES:", sorted(required-found)); sys.exit(1)
failed = df[df["estado"] != "OK"]
if not failed.empty:
    print("HAY CONFIGURACIONES NO COMPLETADAS:")
    print(failed[["metodo","representacion","estado"]].to_string(index=False))
    print("Instala el modelo: python -m spacy download es_core_news_sm; luego vuelve a ejecutar experimentos_nlp.py")
    sys.exit(1)
for col in ["cv_f1_macro_media","test_f1_macro","test_accuracy"]:
    if df[col].isna().any():
        print("MÉTRICAS VACÍAS:", col); sys.exit(1)
ex = pd.read_csv(OUT/"ejemplos_preprocesamiento.csv")
if not {"texto_original","1_normalizacion","2_tokens","3_tokens_sin_stopwords","4_stemming","5_lematizacion","se_conserva_no"}.issubset(ex.columns):
    print("El archivo de ejemplos no contiene todos los pasos NLP."); sys.exit(1)
print("CORRECTO: las seis comparaciones se ejecutaron y los archivos de demostración existen.")
print(df[["metodo","representacion","cv_f1_macro_media","test_f1_macro","test_accuracy","vocabulario"]].to_string(index=False))
