# Contribuciones

## Elizabeth Peña: Datos y NLP
Commit `edecc20` ("Entrega etapa1 parcial de topicos") y revisión/merge del PR #1.
- Dataset: corpus COAH (data/corpus_coah.xml), conversión a CSV con cargar_coah.py y etiqueta de sentimiento derivada de las estrellas.
- Script reproducible preparar_datos.py: validación, eliminación de duplicados, división estratificada train/test y artefactos (preprocesador.joblib, matrices TF-IDF, metadatos.json); verificación con verificar_entrega.py.
- Pipeline de NLP en preprocesamiento.py (normalización, stopwords conservando negaciones e intensificadores, TF-IDF).
- Notebooks eda.ipynb y division_desbalance.ipynb; revisión manual de etiquetas en muestra_etiquetas.xlsx.
- Experimentos de representación (experimentos_nlp.py, demostracion_tecnicas_nlp.py, verificar_experimentos_nlp.py, resultados en resultados_nlp/) y GUIA_TECNICAS_NLP_PASO_A_PASO.md.
- README inicial (estructura, contrato para Modelado y Backend, decisiones y limitaciones).

## Claudia Tapia: Modelado
- Notebook notebooks/modelado.ipynb: línea base, regresión logística y Naive Bayes (Complement) sobre el pipeline de preprocesamiento.py, ajuste de hiperparámetros con validación cruzada y elección del modelo en producción por F1 macro.
- Evaluación final en el conjunto de prueba, matriz de confusión, interpretación del modelo y análisis de errores.
- Artefactos en artefactos/: modelo_final.joblib, metricas_modelado.json y tabla_comparativa_modelos.csv.
- Sección "Modelado (resultados)" del README.
- Uso de IA: se usó un asistente de IA (Claude) como apoyo para estructurar el notebook y redactar borradores; la ejecución y la interpretación de resultados son de la integrante.

## Otros integrantes
(completar con lo que hizo cada uno; debe coincidir con el historial de commits)