"""
src/models/entrenar.py
======================
Script reproducible de entrenamiento: con UN solo comando va desde los datos
crudos hasta los modelos guardados. Hace lo mismo que notebooks/modelado.ipynb
(el notebook conserva las explicaciones, gráficos y análisis de errores):

  1. Ejecuta src/nlp/preparar_datos.py: corpus XML -> data/coah.csv ->
     train/test estratificado -> preprocesador y matrices TF-IDF.
  2. Línea base (clase más frecuente).
  3. Búsqueda de hiperparámetros con validación cruzada estratificada (5
     particiones, SOLO train) para Regresión logística y Naive Bayes (Complement).
  4. Elige el modelo en producción por F1 macro de validación cruzada.
  5. Evalúa UNA vez en test y guarda en artefactos/:
     modelo_final.joblib, metricas_modelado.json y tabla_comparativa_modelos.csv.

Uso, desde la RAÍZ del proyecto:

    python -m src.models.entrenar
"""
import json
import time
import warnings
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             make_scorer, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

from src.nlp import preparar_datos
from src.nlp.preprocesamiento import crear_preprocesador

RAIZ = Path(__file__).resolve().parents[2]       # src/models -> src -> raíz
DATA = RAIZ / "data"
ARTEFACTOS = RAIZ / "artefactos"
SEED = 42
CLASES = ["negativo", "neutro", "positivo"]


def f1_neutro(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=["neutro"], average="macro", zero_division=0)


def recall_neutro(y_true, y_pred):
    return recall_score(y_true, y_pred, labels=["neutro"], average="macro", zero_division=0)


METRICAS = {
    "accuracy": "accuracy",
    "f1_macro": "f1_macro",
    "f1_neutro": make_scorer(f1_neutro),
    "recall_neutro": make_scorer(recall_neutro),
}


def construir(clf) -> Pipeline:
    """Pipeline completo: normalizar -> TF-IDF (src/nlp/preprocesamiento.py) -> clasificador."""
    return Pipeline([("prep", crear_preprocesador()), ("clf", clf)])


def fila(nombre, estimador, X, y, cv) -> dict:
    t0 = time.time()
    r = cross_validate(estimador, X, y, cv=cv, scoring=METRICAS)
    return {
        "Modelo": nombre,
        "Accuracy": r["test_accuracy"].mean(),
        "F1 macro": r["test_f1_macro"].mean(),
        "Desv. F1 macro": r["test_f1_macro"].std(),
        "F1 neutro": r["test_f1_neutro"].mean(),
        "Recall neutro": r["test_recall_neutro"].mean(),
        "Tiempo CV (s)": time.time() - t0,
    }


def main() -> None:
    warnings.filterwarnings("ignore")
    preparar_datos.main()             # datos crudos -> train/test -> artefactos de NLP
    print("\n=== Modelado ===")
    train, test = pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")
    X_train, y_train = train["texto"], train["sentimiento"]
    X_test, y_test = test["texto"], test["sentimiento"]
    print(f"Entrenamiento: {len(train)} | Prueba: {len(test)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    busquedas = {
        "Regresión logística": (
            construir(LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
            {"prep__tfidf__ngram_range": [(1, 1), (1, 2)],
             "prep__tfidf__min_df": [1, 2, 3],
             "clf__C": [0.1, 1, 10]},
        ),
        "Naive Bayes (Complement)": (
            construir(ComplementNB()),
            {"prep__tfidf__ngram_range": [(1, 1), (1, 2)],
             "prep__tfidf__min_df": [1, 2, 3],
             "clf__alpha": [0.1, 0.5, 1.0]},
        ),
    }

    mejores, parametros = {}, {}
    for nombre, (pipe, grid) in busquedas.items():
        gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=1, refit=True)
        gs.fit(X_train, y_train)
        mejores[nombre] = gs.best_estimator_
        parametros[nombre] = {k: str(v) for k, v in gs.best_params_.items()}
        print(f"{nombre}: {gs.best_params_} | F1 macro (CV) {gs.best_score_:.4f}")

    filas = [fila("Línea base", DummyClassifier(strategy="most_frequent"), X_train, y_train, cv)]
    filas += [fila(nombre, est, X_train, y_train, cv) for nombre, est in mejores.items()]
    tabla = pd.DataFrame(filas).set_index("Modelo").round(4)
    nombre_final = tabla.drop(index="Línea base")["F1 macro"].idxmax()
    print(tabla.to_string())
    print(f"\nModelo en producción: {nombre_final}")

    # Evaluación única en test
    modelo_final = mejores[nombre_final]
    pred = modelo_final.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1m = f1_score(y_test, pred, average="macro")
    print(f"Prueba -> accuracy: {acc:.4f} | F1 macro: {f1m:.4f}")
    print(classification_report(y_test, pred, labels=CLASES, digits=3, zero_division=0))

    ARTEFACTOS.mkdir(exist_ok=True)
    joblib.dump(modelo_final, ARTEFACTOS / "modelo_final.joblib")
    reporte = classification_report(y_test, pred, labels=CLASES, output_dict=True, zero_division=0)
    metricas = {
        "modelo_en_produccion": nombre_final,
        "mejores_parametros": parametros,
        "validacion_cruzada": tabla.reset_index().to_dict(orient="records"),
        "prueba": {
            "accuracy": round(float(acc), 4),
            "f1_macro": round(float(f1m), 4),
            "por_clase": {c: {k: round(float(v), 4) for k, v in reporte[c].items()} for c in CLASES},
        },
        "clases": CLASES,
        "semilla": SEED,
        "version_scikit_learn": sklearn.__version__,
    }
    (ARTEFACTOS / "metricas_modelado.json").write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")
    tabla.to_csv(ARTEFACTOS / "tabla_comparativa_modelos.csv")
    print("Guardado en artefactos/: modelo_final.joblib, metricas_modelado.json, tabla_comparativa_modelos.csv")


if __name__ == "__main__":
    main()
