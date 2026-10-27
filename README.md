# Análisis de sentimiento en reseñas de hoteles — Parcial 1

Tópicos Avanzados de Programación · UAGRM · Gestión 2-2026

Este repositorio contiene la entrega del rol **Datos y NLP** (dataset, análisis
exploratorio y pipeline de preprocesamiento). El rol **Modelado** (entrenamiento,
ajuste y comparación de modelos) continúa a partir de aquí.

```
Datos  ->  Pipeline de NLP  ->  Modelos  ->  Artefactos  ->  Backend  ->  Frontend
└──── esta entrega ────┘      (Modelado)
```

## Estructura

```
cargar_coah.py            XML original -> data/coah.csv (+ etiqueta sentimiento)
preprocesamiento.py       Pipeline de NLP: normalización + TF-IDF   <-- compartido con el backend
preparar_datos.py         SCRIPT REPRODUCIBLE: datos crudos -> train/test -> artefactos
verificar_entrega.py      Comprueba que la entrega está completa y consistente
requirements.txt          Dependencias
data/
  corpus_coah.xml         Corpus original (SINAI, Universidad de Jaén)
  coah.csv                Dataset completo (1.816 reseñas)
  train.csv               Entrenamiento (1.433 reseñas, sin duplicados)
  test.csv                Prueba (359 reseñas) — BAJO LLAVE
artefactos/
  preprocesador.joblib    Pipeline NLP ya ajustado (con train)
  X_train_tfidf.npz       Matriz TF-IDF de entrenamiento (comodidad)
  X_test_tfidf.npz        Matriz TF-IDF de prueba (comodidad)
  metadatos.json          Clases, semilla, tamaños, versiones de librerías
notebooks/
  eda.ipynb               Análisis exploratorio
  division_desbalance.ipynb  Duplicados, división, desbalance (prueba de concepto)
muestra_etiquetas.xlsx    Revisión manual de 60 reseñas (90 % de coincidencia)
```

## Cómo reproducir todo

```bash
pip install -r requirements.txt
python preparar_datos.py        # genera train/test y los artefactos
python verificar_entrega.py     # comprueba que todo quedó bien
```

Ejecutar siempre desde la carpeta del proyecto. Con los mismos datos y versiones
de librerías, el resultado es idéntico en cada ejecución (semilla fija = 42).

## Para Modelado (contrato de entrega)

| Qué | Dónde | Notas |
|---|---|---|
| Datos de entrenamiento | `data/train.csv` | Columnas: `id, rank, titulo, texto, sentimiento`. Entrada: `texto`. Etiqueta: `sentimiento`. |
| Datos de prueba | `data/test.csv` | **No usar para ajustar ni elegir modelos.** Solo una evaluación final. |
| Clases (orden fijo) | `artefactos/metadatos.json` | `negativo`, `neutro`, `positivo` |
| Pipeline NLP | `preprocesamiento.crear_preprocesador()` | Normaliza el texto y aplica TF-IDF. |
| Pipeline NLP ya ajustado | `artefactos/preprocesador.joblib` | Ajustado solo con train. |
| Matrices TF-IDF | `artefactos/X_*_tfidf.npz` | Solo por comodidad. |

Puntos a tener en cuenta:

- **Desbalance:** positivo 56 %, negativo 28 %, neutro 16 %. En la prueba de
  concepto, `class_weight='balanced'` subió el F1 macro de 0,590 a 0,714 y el
  recall de la clase neutra de 0,057 a 0,461 (regresión logística). Mirar
  siempre F1 macro y el rendimiento en la clase **neutro**, no solo la exactitud.
- **Validación cruzada:** hacerla sobre `train.csv` (p. ej. `StratifiedKFold`),
  nunca sobre `test.csv`.
- **Sin fuga de información:** al validar, meter el preprocesador DENTRO del
  `Pipeline` (así el TF-IDF se reajusta en cada pliegue).

### Ejemplo mínimo para empezar

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from preprocesamiento import crear_preprocesador

train = pd.read_csv("data/train.csv")
X, y = train["texto"], train["sentimiento"]

modelo = Pipeline(crear_preprocesador().steps + [
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(cross_val_score(modelo, X, y, cv=cv, scoring="f1_macro").mean())
```

### Qué debe entregar Modelado para el backend

Un **único** pipeline completo (preprocesamiento + mejor modelo) guardado con
`joblib.dump(modelo, "artefactos/modelo.joblib")`, de modo que el backend solo
haga `modelo.predict([texto])`.

## Para el Backend

- Hace falta `preprocesamiento.py` en el proyecto del backend: el archivo
  `.joblib` guarda una referencia a la clase `TextoNormalizador`.
- Usar la **misma versión de scikit-learn** que figura en `artefactos/metadatos.json`.
- El pipeline acepta texto crudo tal como lo escribe el usuario; el backend
  debería rechazar textos vacíos antes de predecir.

## Decisiones de preprocesamiento

- Minúsculas, sin acentos (se conserva la ñ), sin URLs, números ni signos.
- Alargamientos reducidos (`buenooo` → `bueno`).
- Stopwords en español, **conservando negaciones e intensificadores** (`no`, `sin`,
  `nada`, `nunca`, `muy`, `poco`, `pero`...), porque cambian el sentimiento.
- TF-IDF con unigramas y bigramas (`no recomiendo`), `min_df=2`, `sublinear_tf=True`.

## Limitaciones conocidas

- La etiqueta se deriva de las estrellas (1-2 negativo, 3 neutro, 4-5 positivo).
  En la revisión manual, la clase neutra coincidió en solo un 80 % de los casos.
- El corpus es de reseñas de hoteles de Granada (España): el modelo puede
  funcionar peor con otros dominios o con español de otras regiones.
