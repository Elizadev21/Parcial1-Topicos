"""
tests/verificar_entrega.py
==========================
Comprueba que la entrega "Datos y NLP" está completa y consistente.
Se ejecuta DESPUÉS de src/nlp/preparar_datos.py, desde la raíz del proyecto:

    python -m tests.verificar_entrega

No evalúa ningún modelo ni toca la partición de prueba para medir rendimiento
(esa evaluación es responsabilidad del rol de Modelado).
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

BASE = Path(__file__).resolve().parents[1]  # tests -> raíz del proyecto
DATA, ART = BASE / "data", BASE / "artefactos"

ARCHIVOS = [
    DATA / "train.csv", DATA / "test.csv",
    ART / "preprocesador.joblib", ART / "X_train_tfidf.npz",
    ART / "X_test_tfidf.npz", ART / "metadatos.json",
]


def comprobar(condicion: bool, mensaje: str) -> None:
    print(("  OK   " if condicion else "  FALLA ") + mensaje)
    if not condicion:
        raise SystemExit(1)


def main() -> None:
    print("Verificando la entrega...")
    for a in ARCHIVOS:
        comprobar(a.exists(), f"existe {a.relative_to(BASE)}")

    meta = json.loads((ART / "metadatos.json").read_text(encoding="utf-8"))
    train, test = pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")
    comprobar(len(train) == meta["n_train"] and len(test) == meta["n_test"],
              f"tamaños coinciden con metadatos ({len(train)} train / {len(test)} test)")
    comprobar(set(train["texto"]).isdisjoint(set(test["texto"])),
              "ningún texto está en entrenamiento y prueba a la vez")
    comprobar(not train["texto"].duplicated().any(), "sin duplicados en entrenamiento")
    comprobar(sorted(train["sentimiento"].unique()) == meta["clases"],
              f"clases presentes: {meta['clases']}")

    # El preprocesador guardado debe reproducir exactamente la matriz guardada.
    pre = joblib.load(ART / "preprocesador.joblib")
    X_train_guardada = sp.load_npz(ART / "X_train_tfidf.npz")
    X_nueva = pre.transform(train["texto"])
    comprobar(X_nueva.shape == X_train_guardada.shape, f"forma de X_train {X_nueva.shape}")
    comprobar(np.allclose(X_nueva.toarray(), X_train_guardada.toarray()),
              "el preprocesador cargado reproduce X_train exactamente")

    # Caso de uso del backend: un solo texto crudo.
    ejemplo = pre.transform(["El hotel estaba sucio y el personal fue muy grosero."])
    comprobar(ejemplo.shape == (1, X_nueva.shape[1]) and ejemplo.nnz > 0,
              "un texto nuevo (como lo enviaría el backend) se transforma bien")

    print("\nTodo correcto: la entrega está lista para Modelado.")


if __name__ == "__main__":
    main()
