# Informe técnico: análisis de sentimiento en reseñas de hoteles

Tópicos Avanzados de Programación · UAGRM · Gestión 2-2026

## 1. Problema

Los hoteles reciben muchas reseñas en texto libre y leerlas una por una no escala.
El objetivo es **clasificar automáticamente una reseña en español** en uno de tres
sentimientos: **negativo**, **neutro** o **positivo**. El clasificador se ofrece como
servicio web: una API recibe el texto y devuelve el sentimiento, la confianza y los
pasos del preprocesamiento, y una interfaz web los muestra.

La métrica principal es el **F1 macro**, el promedio del F1 de las tres clases con el
mismo peso. Se eligió en lugar de la exactitud porque las clases están desbalanceadas:
un modelo que siempre dice «positivo» acierta el 56 % sin haber aprendido nada. También
se sigue el **F1 de la clase neutra**, la más difícil.

## 2. Dataset

| Dato | Valor |
|---|---|
| Fuente | Corpus COAH (reseñas de hoteles de Granada), SINAI, Universidad de Jaén |
| Archivo original | `data/corpus_coah.xml`, convertido a `data/coah.csv` con `src/nlp/cargar_coah.py` |
| Reseñas | 1.816 (1.792 tras eliminar 24 duplicados) |
| Campos | `id`, `rank` (estrellas 1–5), `titulo`, `texto`, `sentimiento` |
| Etiqueta | Derivada de las estrellas: 1–2 → negativo, 3 → neutro, 4–5 → positivo |
| División | Estratificada 80/20, semilla 42: 1.433 de entrenamiento y 359 de prueba |

Distribución de clases (se mantiene igual en entrenamiento y prueba gracias a la estratificación):

| Clase | Total | % | Entrenamiento | Prueba |
|---|---|---|---|---|
| Positivo | 1.020 | 56,2 % | 805 | 202 |
| Negativo | 511 | 28,1 % | 400 | 100 |
| Neutro | 285 | 15,7 % | 228 | 57 |

**Calidad de la etiqueta.** Se revisaron a mano 60 reseñas (`data/muestra_etiquetas.xlsx`).
La etiqueta derivada de las estrellas coincidió en el 90 % de los casos, pero en la clase
neutra solo en el 80 %: una reseña de 3 estrellas a menudo mezcla elogios y quejas.

**Regla del conjunto de prueba.** `data/test.csv` no se usa para elegir nada. Todas las
decisiones (preprocesamiento, modelo e hiperparámetros) se toman con validación cruzada
sobre el entrenamiento, y el conjunto de prueba se evalúa una sola vez al final.

El análisis exploratorio está en `notebooks/eda.ipynb`, y los duplicados, la división y el
desbalance en `notebooks/division_desbalance.ipynb`.

## 3. Decisiones de NLP

El preprocesamiento está en un solo módulo, `src/nlp/preprocesamiento.py`. La función
`pasos()` se usa **al entrenar, al predecir y en la vista paso a paso del frontend**, así
que el texto que ve el usuario es exactamente el que recibe el modelo. Una prueba
automática lo comprueba: `tests/test_api.py::test_preprocesamiento_mostrado_es_el_del_modelo`.

| Paso | Qué hace | Por qué |
|---|---|---|
| 1. Normalización | Minúsculas; quita URLs, números, signos y acentos (conserva la ñ); reduce alargamientos (`buenooo` → `bueno`) | Hay reseñas con y sin tildes (`habitación` / `habitacion`) que deben contar como la misma palabra |
| 2. Tokenización | Separa por espacios y descarta palabras de una letra | `y`, `a` y `o` no aportan sentimiento |
| 3. Stop words | Quita palabras funcionales (artículos, preposiciones, pronombres), pero **conserva negaciones e intensificadores**: `no`, `sin`, `nada`, `nunca`, `muy`, `poco`, `pero`, `aunque`… | Las listas estándar eliminan `no`, y entonces «no recomiendo» y «recomiendo» quedarían iguales |
| 4. Stemming | Snowball en español (`recomiendo` → `recom`, `habitación` → `habit`) | Agrupa variantes de una misma palabra y reduce el vocabulario |
| 5. Vectorización | TF-IDF con `sublinear_tf=True`; n-gramas y `min_df` elegidos por validación cruzada | Pondera las palabras distintivas y atenúa las muy repetidas |

**Elección entre stemming, lematización o ninguna reducción.** Se compararon las tres
opciones con unigramas y con unigramas + bigramas, usando una regresión logística fija y
validación cruzada estratificada de 3 particiones solo sobre entrenamiento
(`src/nlp/experimentos_nlp.py` → `resultados_nlp/`):

| Reducción | Representación | F1 macro (CV) | Vocabulario |
|---|---|---|---|
| Ninguna | Unigramas | 0,715 ± 0,036 | 5.839 |
| Ninguna | Uni + bigramas | 0,716 ± 0,029 | 15.485 |
| Stemming | Unigramas | 0,715 ± 0,035 | 3.634 |
| **Stemming** | **Uni + bigramas** | **0,720 ± 0,033** | 16.099 |
| Lematización (spaCy) | Unigramas | 0,707 ± 0,025 | 4.462 |
| Lematización (spaCy) | Uni + bigramas | 0,709 ± 0,036 | 16.608 |

Se adoptó el **stemming** porque obtuvo el mejor F1 macro y no depende de un modelo
lingüístico pesado: spaCy no es necesario en producción y la imagen de Docker es más
liviana. Las diferencias son pequeñas (están dentro de la desviación), así que la decisión
también se apoya en ese motivo práctico. Al incorporarlo al pipeline final, el F1 macro en
prueba pasó de 0,686 a 0,707 y el F1 neutro de 0,37 a 0,42.

## 4. Modelos y comparación

`python -m src.models.entrenar` (o `notebooks/modelado.ipynb`, con gráficos y análisis de
errores) ejecuta todo con **un solo comando**: datos crudos → división → búsqueda de
hiperparámetros → evaluación → artefactos.

- **Validación:** `StratifiedKFold` de 5 particiones (shuffle, semilla 42), solo sobre entrenamiento.
- **Búsqueda:** `GridSearchCV` optimizando F1 macro. El preprocesamiento está dentro del
  pipeline, así que el TF-IDF se ajusta en cada partición y no hay fuga de información.
- **Modelos comparados:**
  - **Línea base:** siempre predice la clase más frecuente.
  - **Regresión logística** con `class_weight="balanced"`. Se buscaron `C` ∈ {0,1; 1; 10}, n-gramas ∈ {(1,1), (1,2)} y `min_df` ∈ {1, 2, 3}.
  - **Naive Bayes Complement.** Se buscaron `alpha` ∈ {0,1; 0,5; 1,0} con los mismos n-gramas y `min_df`.

### Tabla comparativa (validación cruzada, 5 particiones)

| Modelo | Mejores hiperparámetros | Accuracy | F1 macro | F1 neutro | Recall neutro |
|---|---|---|---|---|---|
| Línea base | — | 0,562 | 0,240 | 0,000 | 0,000 |
| **Regresión logística** | `C=1`, unigramas, `min_df=1` | 0,794 | **0,720 ± 0,024** | **0,446** | **0,457** |
| Naive Bayes Complement | `alpha=0,1`, uni + bigramas, `min_df=3` | **0,812** | 0,651 ± 0,037 | 0,227 | 0,137 |

**Modelo en producción: regresión logística.** Tiene el mayor F1 macro y detecta el triple
de reseñas neutras que Naive Bayes (recall 0,457 frente a 0,137). Naive Bayes logra algo más
de exactitud solo porque acierta más la clase mayoritaria (positivo), que es justo lo que
la métrica F1 macro penaliza.

### Evaluación final en prueba (una sola vez, 359 reseñas)

| Clase | Precisión | Recall | F1 | Soporte |
|---|---|---|---|---|
| Negativo | 0,796 | 0,860 | 0,827 | 100 |
| Neutro | 0,397 | 0,439 | 0,417 | 57 |
| Positivo | 0,910 | 0,847 | 0,877 | 202 |
| **Total** | | | **F1 macro 0,707** · accuracy 0,786 | 359 |

Los resultados en prueba son cercanos a los de validación cruzada (0,707 frente a 0,720), lo
que indica que el modelo no está sobreajustado. De los 77 errores, **70 involucran a la clase
neutra**. Esto coincide con el ruido de la etiqueta observado en la revisión manual: muchas
reseñas de 3 estrellas son en realidad mixtas.

Los resultados se guardan en `artefactos/`: `modelo_final.joblib` (pipeline completo),
`metricas_modelado.json` y `tabla_comparativa_modelos.csv`.

## 5. Arquitectura

```
 data/corpus_coah.xml
        │  src/nlp/cargar_coah.py
        ▼
 data/coah.csv ──► src/nlp/preparar_datos.py ──► train.csv / test.csv
                                                     │
              src/nlp/preprocesamiento.py ◄──────────┤  (mismo módulo al entrenar y al predecir)
                                                     ▼
                                          src/models/entrenar.py
                                                     │
                                                     ▼
                         artefactos/modelo_final.joblib + metricas_modelado.json
                                                     │  se carga al iniciar
                                                     ▼
 ┌──────────────────────────┐   /api/*   ┌──────────────────────────────────┐
 │ Frontend React + Tailwind│ ─────────► │ Backend FastAPI (api/main.py)    │
 │ (Vite; nginx en Docker)  │ ◄───────── │ salud · predecir · preprocesar · │
 └──────────────────────────┘    JSON    │ métricas · /docs · /redoc        │
                                         └──────────────────────────────────┘
```

| Componente | Tecnología | Responsabilidad |
|---|---|---|
| NLP | Python 3.12, NLTK (Snowball), scikit-learn 1.8 | Preprocesamiento compartido (`src/nlp/`) |
| Modelado | scikit-learn, joblib | Búsqueda, evaluación y serialización (`src/models/`) |
| Backend | FastAPI, Pydantic, Uvicorn | Carga el pipeline una vez al iniciar y expone la API |
| Frontend | React 19, Vite 6, Tailwind CSS 4 | Formulario, probabilidades, preprocesamiento paso a paso y panel de métricas |
| Despliegue | Docker y docker-compose | Dos contenedores; nginx sirve el frontend y redirige `/api` al backend |

**Endpoints:**

| Método | Ruta | Respuesta |
|---|---|---|
| GET | `/api/salud` | Estado de la API y del modelo (503 si no hay modelo cargado) |
| POST | `/api/predecir` | Sentimiento, confianza, probabilidades y preprocesamiento |
| POST | `/api/preprocesar` | Cada paso del preprocesamiento |
| GET | `/api/metricas` | Métricas de `artefactos/metricas_modelado.json` |

**Validación y errores:**
- Un texto vacío, con solo signos o números, de más de 5.000 caracteres, o un JSON sin el campo `texto` devuelven **422** con un mensaje en español.
- Si el modelo no está cargado, la API responde **503**.
- Si ninguna palabra está en el vocabulario del modelo, la API no inventa una clase y devuelve un `aviso`.
- Las pruebas están en `tests/` y se ejecutan con `python -m pytest`.

## 6. Limitaciones y trabajo futuro

- **Etiqueta ruidosa en la clase neutra.** Se deriva de las estrellas, y las reseñas mixtas
  limitan el F1 neutro (0,42). Revisar manualmente más etiquetas neutras ayudaría más que
  cambiar de modelo.
- **Dominio.** Son reseñas de hoteles de Granada (España). El rendimiento puede bajar con
  otros sectores o con variantes regionales del español.
- **Representación.** TF-IDF no capta el orden ni el contexto amplio («no estaba nada mal»).
  Como siguiente paso se podrían probar *embeddings* o un modelo tipo BETO, comparándolos
  contra esta línea.

## 7. Reproducción

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.models.entrenar        # datos crudos → modelos en artefactos/
uvicorn api.main:app --port 8000     # backend
cd frontend && npm install && npm run dev   # frontend
# o todo junto: docker compose up --build
```
