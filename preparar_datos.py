"""
preparar_datos.py
=================
Script REPRODUCIBLE del rol "Datos y NLP".

Ejecutándolo (un solo comando, desde la carpeta del proyecto):

    python preparar_datos.py

recorre todo el camino  Datos -> Pipeline de NLP -> Artefactos:

  1. Carga data/coah.csv
  2. Valida el dataset (columnas, faltantes, etiquetas válidas).
  3. Elimina textos duplicados (1.816 -> 1.792 reseñas).
  4. Divide 80 % entrenamiento / 20 % prueba, estratificado, random_state=42.
  5. Ajusta el pipeline de NLP (normalización + TF-IDF) SOLO con entrenamiento
     (así no hay fuga de información desde la prueba).
  6. Guarda todo en data/ y artefactos/ 

Con los mismos datos y las mismas versiones de librerías, el resultado es
siempre idéntico (no hay nada aleatorio sin semilla fija).
"""
import json
import sys
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import scipy.sparse as sp
import sklearn
from sklearn.model_selection import train_test_split

from preprocesamiento import crear_preprocesador

# ----------------------------- Configuración -------------------------------
BASE = Path(__file__).resolve().parent          # carpeta del proyecto
DATA = BASE / "data"
ARTEFACTOS = BASE / "artefactos"

XML_CRUDO = DATA / "corpus_coah.xml"
CSV_COMPLETO = DATA / "coah.csv"

SEMILLA = 42
TAMANO_PRUEBA = 0.20
CLASES = ["negativo", "neutro", "positivo"]      # orden fijo de las clases
COLUMNAS = ["id", "rank", "titulo", "texto", "sentimiento"]


def cargar_dataset() -> pd.DataFrame:
    """Paso 1 y 2: carga el CSV (regenerándolo desde el XML si hace falta) y valida."""
    if not CSV_COMPLETO.exists():
        print(f"[1] {CSV_COMPLETO.name} no existe: se genera desde {XML_CRUDO.name}")
        from cargar_coah import cargar
        cargar(str(XML_CRUDO)).to_csv(CSV_COMPLETO, index=False)
    df = pd.read_csv(CSV_COMPLETO)
    print(f"[1] Dataset cargado: {len(df)} registros, columnas {list(df.columns)}")

    # Validaciones: si algo está mal, el script se detiene con un mensaje claro.
    assert list(df.columns) == COLUMNAS, f"Columnas inesperadas: {list(df.columns)}"
    assert df.isna().sum().sum() == 0, "Hay valores faltantes en el dataset"
    assert set(df["sentimiento"]) == set(CLASES), "Etiquetas de sentimiento inesperadas"
    assert (df["texto"].str.strip() != "").all(), "Hay reseñas con texto vacío"
    print("[2] Validación OK: sin faltantes, sin textos vacíos, 3 clases esperadas")
    return df


def quitar_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 3: elimina reseñas con texto repetido (se conserva la primera)."""
    n_antes = len(df)
    df = df.drop_duplicates(subset="texto").reset_index(drop=True)
    print(f"[3] Duplicados eliminados: {n_antes - len(df)} ({n_antes} -> {len(df)})")
    return df


def dividir(df: pd.DataFrame):
    """Paso 4: división estratificada con semilla fija."""
    train, test = train_test_split(
        df, test_size=TAMANO_PRUEBA, stratify=df["sentimiento"], random_state=SEMILLA
    )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    # Comprobaciones de calidad de la división
    assert set(train["texto"]).isdisjoint(set(test["texto"])), \
        "Hay textos repetidos entre entrenamiento y prueba"
    print(f"[4] División: entrenamiento={len(train)} | prueba={len(test)}")
    prop = pd.DataFrame({
        "completo": df["sentimiento"].value_counts(normalize=True),
        "train": train["sentimiento"].value_counts(normalize=True),
        "test": test["sentimiento"].value_counts(normalize=True),
    }).round(3).loc[CLASES]
    print("    Proporciones por clase:\n" + prop.to_string().replace("\n", "\n    "))
    return train, test


def main() -> None:
    DATA.mkdir(exist_ok=True)
    ARTEFACTOS.mkdir(exist_ok=True)

    df = quitar_duplicados(cargar_dataset())
    train, test = dividir(df)

    # Se guardan las particiones para que Modelado use EXACTAMENTE la misma.
    train.to_csv(DATA / "train.csv", index=False)
    test.to_csv(DATA / "test.csv", index=False)
    print("    Guardados: data/train.csv y data/test.csv")

    # Paso 5: pipeline de NLP ajustado solo con entrenamiento.
    pre = crear_preprocesador()
    X_train = pre.fit_transform(train["texto"])
    X_test = pre.transform(test["texto"])       # transform, NO fit
    n_terminos = len(pre.named_steps["tfidf"].vocabulary_)
    print(f"[5] TF-IDF ajustado con entrenamiento: {n_terminos} términos | "
          f"X_train {X_train.shape} | X_test {X_test.shape}")

    # Paso 6: artefactos
    joblib.dump(pre, ARTEFACTOS / "preprocesador.joblib")
    sp.save_npz(ARTEFACTOS / "X_train_tfidf.npz", X_train)
    sp.save_npz(ARTEFACTOS / "X_test_tfidf.npz", X_test)
    metadatos = {
        "fecha_generacion": date.today().isoformat(),
        "clases": CLASES,
        "semilla": SEMILLA,
        "tamano_prueba": TAMANO_PRUEBA,
        "registros_originales": int(len(pd.read_csv(CSV_COMPLETO))),
        "registros_sin_duplicados": int(len(df)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_terminos_tfidf": int(n_terminos),
        "version_scikit_learn": sklearn.__version__,
        "version_pandas": pd.__version__,
        "version_python": sys.version.split()[0],
    }
    (ARTEFACTOS / "metadatos.json").write_text(
        json.dumps(metadatos, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[6] Artefactos guardados en artefactos/: preprocesador.joblib, "
          "X_train_tfidf.npz, X_test_tfidf.npz, metadatos.json")
    print("\nListo. Siguiente paso (Modelado): ver README.md, sección 'Para Modelado'.")


if __name__ == "__main__":
    main()
