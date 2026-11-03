# Contribuciones

## Elizabeth Peña: Datos y NLP
Commit `edecc20` ("Entrega etapa1 parcial de topicos") y revisión/merge del PR #1.
- Dataset: corpus COAH (data/corpus_coah.xml), conversión a CSV con src/nlp/cargar_coah.py y etiqueta de sentimiento derivada de las estrellas.
- Script reproducible src/nlp/preparar_datos.py: validación, eliminación de duplicados, división estratificada train/test y artefactos (preprocesador.joblib, matrices TF-IDF, metadatos.json); verificación con tests/verificar_entrega.py.
- Pipeline de NLP en src/nlp/preprocesamiento.py (normalización, stopwords conservando negaciones e intensificadores, TF-IDF).
- Notebooks eda.ipynb y division_desbalance.ipynb; revisión manual de etiquetas en data/muestra_etiquetas.xlsx.
- Experimentos de representación (src/nlp/experimentos_nlp.py, src/nlp/demostracion_tecnicas_nlp.py, tests/verificar_experimentos_nlp.py, resultados en resultados_nlp/) y docs/GUIA_TECNICAS_NLP_PASO_A_PASO.md.
- README inicial (estructura, contrato para Modelado y Backend, decisiones y limitaciones).

## Claudia Tapia: Modelado
- Notebook notebooks/modelado.ipynb: línea base, regresión logística y Naive Bayes (Complement) sobre el pipeline de src/nlp/preprocesamiento.py, ajuste de hiperparámetros con validación cruzada y elección del modelo en producción por F1 macro.
- Evaluación final en el conjunto de prueba, matriz de confusión, interpretación del modelo y análisis de errores.
- Artefactos en artefactos/: modelo_final.joblib, metricas_modelado.json y tabla_comparativa_modelos.csv.
- Sección "Modelado (resultados)" del README.
- Uso de IA: se usó un asistente de IA (Claude) como apoyo para estructurar el notebook y redactar borradores; la ejecución y la interpretación de resultados son de la integrante.

## Sergio David Cambara Dorbigny: Backend, Frontend e integración
- NLP y ML:
  - Stemming Snowball añadido al pipeline de src/nlp/preprocesamiento.py, elegido por F1 macro en validación cruzada (resultados_nlp/). La función `pasos()` comparte el mismo preprocesamiento entre entrenamiento, API y frontend.
  - Script src/models/entrenar.py: con un solo comando va desde el corpus XML hasta los modelos guardados (mismo proceso que notebooks/modelado.ipynb).
  - Reentrenamiento y nueva ejecución del notebook: F1 macro en prueba de 0,686 a 0,707 y F1 de la clase neutra de 0,37 a 0,42.
- Backend (api/main.py), API REST con FastAPI:
  - Endpoints /api/salud, /api/predecir, /api/preprocesar y /api/metricas.
  - Validación de entradas con mensajes de error claros en español.
  - Aviso cuando ninguna palabra de la reseña está en el vocabulario.
  - Documentación en /docs y /redoc (ReDoc fijado a la versión 2 estable).
- Pruebas automatizadas en tests/:
  - test_api.py: endpoints, errores, documentación y que el preprocesamiento mostrado sea el mismo que usa el modelo.
  - test_entrega.py: verificación de la entrega de Datos y NLP.
- Frontend (frontend/), React 19 + Tailwind CSS 4:
  - Menú lateral con botón de hamburguesa.
  - Predicción con probabilidades.
  - Vista del preprocesamiento paso a paso.
  - Panel de métricas y comparación de modelos.
- Contenedores Docker:
  - api/Dockerfile, frontend/Dockerfile (con nginx) y docker-compose.yml.
  - Variables PORT y BACKEND_URL para desplegar en la nube.

