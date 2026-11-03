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
data/                     Datos crudos y procesados
  corpus_coah.xml         Corpus original (SINAI, Universidad de Jaén)
  coah.csv                Dataset completo (1.816 reseñas)
  train.csv               Entrenamiento (1.433 reseñas, sin duplicados)
  test.csv                Prueba (359 reseñas) — BAJO LLAVE
  muestra_etiquetas.xlsx  Revisión manual de 60 reseñas (90 % de coincidencia)
notebooks/                Análisis exploratorio y experimentos
  eda.ipynb               Análisis exploratorio
  division_desbalance.ipynb  Duplicados, división, desbalance (prueba de concepto)
  modelado.ipynb          Modelado explicado paso a paso (gráficos y análisis de errores)
src/
  nlp/                    Pipeline de preprocesamiento
    preprocesamiento.py   Normalización + TF-IDF   <-- compartido con la API
    cargar_coah.py        XML original -> data/coah.csv (+ etiqueta sentimiento)
    preparar_datos.py     SCRIPT REPRODUCIBLE: datos crudos -> train/test -> artefactos
    experimentos_nlp.py   Compara texto normalizado, stemming y lematización
    demostracion_tecnicas_nlp.py  Ejemplos paso a paso de cada técnica
  models/                 Entrenamiento y evaluación
    entrenar.py           Script de un solo comando: datos crudos -> modelos guardados
api/                      Backend FastAPI (salud, predecir, preprocesar, métricas) y su Dockerfile
frontend/                 Interfaz web React + Tailwind CSS (Vite) y su Dockerfile con nginx
tests/                    Pruebas (pytest) y scripts de verificación
docs/                     Informe técnico (informe_tecnico.md), guía de técnicas NLP y capturas
artefactos/               Salidas: preprocesador, matrices TF-IDF, modelo final y métricas
resultados_nlp/           Salidas de los experimentos de NLP
docker-compose.yml        Levanta API + frontend
requirements.txt          Dependencias
```

Todos los comandos se ejecutan **desde la raíz del proyecto**, como módulos
(`python -m ...`), para que `src` sea importable.

## Cómo reproducir todo

Con **un solo comando** se va desde los datos crudos (`data/corpus_coah.xml`)
hasta los modelos guardados en `artefactos/`:

```bash
pip install -r requirements.txt
python -m src.models.entrenar         # datos -> pipeline NLP -> modelos -> artefactos
python -m pytest                      # pruebas automatizadas
```

`entrenar` ejecuta primero `src/nlp/preparar_datos.py` (también se puede lanzar
solo con `python -m src.nlp.preparar_datos` y comprobar con `python -m tests.verificar_entrega`).

Con los mismos datos y versiones de librerías, el resultado es idéntico en cada
ejecución (semilla fija = 42).

## Para Modelado (contrato de entrega)

| Qué | Dónde | Notas |
|---|---|---|
| Datos de entrenamiento | `data/train.csv` | Columnas: `id, rank, titulo, texto, sentimiento`. Entrada: `texto`. Etiqueta: `sentimiento`. |
| Datos de prueba | `data/test.csv` | **No usar para ajustar ni elegir modelos.** Solo una evaluación final. |
| Clases (orden fijo) | `artefactos/metadatos.json` | `negativo`, `neutro`, `positivo` |
| Pipeline NLP | `src.nlp.preprocesamiento.crear_preprocesador()` | Normaliza el texto y aplica TF-IDF. |
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
from src.nlp.preprocesamiento import crear_preprocesador

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
`joblib.dump(modelo, "artefactos/modelo_final.joblib")`, de modo que el backend solo
haga `modelo.predict([texto])`.

## Para el Backend

- Hace falta `src/nlp/preprocesamiento.py` importable (raíz del proyecto en el
  PYTHONPATH): el archivo `.joblib` guarda una referencia a la clase `TextoNormalizador`.
- Usar la **misma versión de scikit-learn** que figura en `artefactos/metadatos.json`.
- El pipeline acepta texto crudo tal como lo escribe el usuario; el backend
  debería rechazar textos vacíos antes de predecir.

## Aplicación web y Docker

La API carga `artefactos/modelo_final.joblib` y usa `src/nlp/preprocesamiento.py`, por eso
se ejecuta desde la raíz del repositorio. El frontend llama a la API mediante `/api`.

Con Docker (recomendado, también para desplegar en la nube):

```bash
docker compose up --build
```

- Aplicación web: `http://localhost:8081`
- API (Swagger): `http://localhost:8002/docs`

Sin Docker, en dos terminales:

```bash
pip install -r api/requirements-docker.txt httpx pytest
uvicorn api.main:app --port 8000
```

```bash
cd frontend
npm ci
npm run dev
```

El frontend usa **React 19** y **Tailwind CSS 4.1** (plugin `@tailwindcss/vite`; estilos en
`frontend/src/styles.css`). Necesita Node.js 18 o superior. El diseño sigue el estilo de
la plantilla iPortfolio (BootstrapMade): menú lateral oscuro fijo (con botón de hamburguesa
en pantallas pequeñas) y una sola página con secciones —portada, análisis con el
preprocesamiento paso a paso, métricas y funcionamiento—. Para usar una foto de fondo en la
portada, colocarla en `frontend/public/portada-suite.jpg`. Desde el menú se enlaza la
documentación de la API (`/docs` y `/redoc`), que nginx también redirige al backend.

Pruebas: `python -m pytest`.

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/api/salud` | Estado de la API y del modelo (503 si no hay modelo) |
| POST | `/api/predecir` | `{"texto": "..."}` → sentimiento, confianza, probabilidades y preprocesamiento |
| POST | `/api/preprocesar` | `{"texto": "..."}` → texto normalizado, tokens, stop words eliminadas y raíces |
| GET | `/api/metricas` | Métricas de `artefactos/metricas_modelado.json` |

Errores: texto vacío, solo signos o números, más de 5000 caracteres o JSON sin el
campo `texto` responden **422** con un mensaje en español en `detail`; si el modelo
no está cargado, **503**. Si ninguna palabra de la reseña está en el vocabulario,
`/api/predecir` no clasifica y devuelve un `aviso`. La documentación de cada
endpoint está en `/docs` (Swagger) y `/redoc`.

En la nube, cada contenedor respeta la variable `PORT` de la plataforma y el
frontend usa `BACKEND_URL` para saber dónde está la API.

## Decisiones de preprocesamiento

- Minúsculas, sin acentos (se conserva la ñ), sin URLs, números ni signos.
- Alargamientos reducidos (`buenooo` → `bueno`).
- Tokenización por palabras; se descartan las de una sola letra.
- Stopwords en español, **conservando negaciones e intensificadores** (`no`, `sin`,
  `nada`, `nunca`, `muy`, `poco`, `pero`...), porque cambian el sentimiento.
- **Stemming** Snowball en español (`recomiendo` → `recom`). Se eligió frente a la
  lematización y a no reducir porque obtuvo el mejor F1 macro en validación cruzada
  en `resultados_nlp/informe_comparaciones.md`.
- TF-IDF con `sublinear_tf=True`. Por defecto usa unigramas y bigramas (`no recom`) con
  `min_df=2`; el modelo final usa los valores elegidos por validación cruzada (unigramas, `min_df=1`).
- La misma función (`src.nlp.preprocesamiento.pasos`) prepara el texto al entrenar,
  al predecir y en la vista paso a paso del frontend; una prueba automática lo comprueba.

## Limitaciones conocidas

- La etiqueta se deriva de las estrellas (1-2 negativo, 3 neutro, 4-5 positivo).
  En la revisión manual, la clase neutra coincidió en solo un 80 % de los casos.
- El corpus es de reseñas de hoteles de Granada (España): el modelo puede
  funcionar peor con otros dominios o con español de otras regiones.

## Modelado (resultados)

Notebook: `notebooks/modelado.ipynb` (script equivalente: `python -m src.models.entrenar`). Compara dos modelos sobre el pipeline de
`src/nlp/preprocesamiento.py`, con validación cruzada estratificada de 5 particiones sobre `data/train.csv`.

| Modelo | Accuracy (CV) | F1 macro (CV) | F1 neutro (CV) |
|---|---|---|---|
| Línea base (clase más frecuente) | 0,562 | 0,240 | 0,000 |
| Regresión logística | 0,794 | 0,720 | 0,446 |
| Naive Bayes (Complement) | 0,812 | 0,651 | 0,227 |

**Modelo en producción: regresión logística** (`C=1`, unigramas, `min_df=1`,
`class_weight='balanced'`, sobre las raíces del stemming). Mayor F1 macro y mucho
mejor recall en la clase neutra (0,457 frente a 0,137 de Naive Bayes), que es la
más difícil. Naive Bayes tiene algo más de exactitud solo porque acierta más la
clase mayoritaria (positivo).

Evaluación única en `data/test.csv`: accuracy 0,786 y F1 macro 0,707 (línea base: 0,563 y 0,240).
F1 por clase: negativo 0,83, neutro 0,42, positivo 0,88. De los 77 errores, 70 involucran a la clase neutra.

Antes de añadir el stemming al pipeline, el mismo proceso daba F1 macro 0,719 en
validación cruzada y 0,686 en prueba (neutro 0,37).

Artefactos generados por el notebook o el script:
- `artefactos/modelo_final.joblib`: pipeline completo (preprocesamiento + modelo), listo para `modelo.predict([texto])`.
- `artefactos/metricas_modelado.json` y `artefactos/tabla_comparativa_modelos.csv`: métricas para el backend y el informe.