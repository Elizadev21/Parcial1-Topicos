\
# Guía paso a paso: técnicas de NLP

**Datos y NLP** del proyecto.

## 1. Cómo ejecutar

Abre PowerShell o la terminal en esta carpeta (`Parcial1-topicos`) y ejecuta:

```bash
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install -r requirements.txt
python -m spacy download es_core_news_sm
python preparar_datos.py
python experimentos_nlp.py
python verificar_experimentos_nlp.py
```

En Windows, si ya existe el entorno virtual, no necesitas crearlo de nuevo. El paso de descargar `es_core_news_sm` es necesario para la lematización.

## 2. Qué se genera

Al ejecutar `experimentos_nlp.py` se crea la carpeta `resultados_nlp/` con:

- `ejemplos_preprocesamiento.csv`: ejemplo por ejemplo, desde el texto original hasta los tokens, stopwords, stemming y lematización.
- `comparacion_modelos_nlp.csv`: resultados numéricos de las seis combinaciones comparadas.
- `informe_comparaciones.md`: explicación del experimento, parámetros, métricas y resultados por clase.
- `detalle_tfidf.csv`: términos aprendidos por TF-IDF y su IDF, como evidencia de la vectorización.

## 3. Las técnicas:

### 3.1 Normalización

**Qué hace:** convierte el texto a minúsculas, quita URL, números, signos y tildes, y reduce letras repetidas exageradamente.

**Ejemplo:** `¡¡NO recomiendo https://ejemplo.com en 2024!!!` → `no recomiendo`.

**Por qué:** variantes como `Hotel` y `hotel` no deberían contarse como palabras diferentes. Se conserva la `ñ`. Quitar números y URL reduce ruido para este experimento; si el número o enlace fuera relevante para la tarea, esa decisión debería revisarse.

### 3.2 Tokenización

**Qué hace:** separa el texto en unidades llamadas tokens, normalmente palabras.

**Ejemplo:** `no recomiendo este hotel` → `['no', 'recomiendo', 'este', 'hotel']`.

En el pipeline, `TfidfVectorizer` de scikit-learn tokeniza automáticamente usando su patrón de palabras. El archivo de ejemplos muestra además los tokens explícitos para que se pueda comprobar visualmente.

### 3.3 Stop words en español

**Qué hace:** quita palabras muy comunes que suelen aportar poco, como artículos o algunas preposiciones.

**Decisión importante:** no se elimina `no`, porque `me gusta` y `no me gusta` significan cosas opuestas. También se preservan `sin`, `nada`, `nunca`, `jamás`, `tampoco`, `muy`, `poco`, `pero` y otras palabras que pueden cambiar el sentimiento. La lista exacta está en `preprocesamiento.py`, en `STOPWORDS_ES`.

El CSV de ejemplos muestra los tokens antes y después de quitar stopwords. La lista se aplica después de normalizar el texto.

### 3.4 Stemming y lematización

**Stemming:** recorta el final de una palabra con reglas; puede producir una raíz que no sea una palabra normal.

**Lematización:** intenta convertir la palabra a su forma de diccionario usando un modelo lingüístico en español.

Se comparan las tres opciones (`ninguno`, `stemming`, `lematizacion`) con dos representaciones TF-IDF. La comparación numérica usa el mismo clasificador de referencia y los mismos datos para que el efecto principal sea la representación del texto.

### 3.5 N-gramas

- **Unigramas `(1,1)`:** considera cada palabra por separado: `no`, `recomiendo`, `hotel`.
- **Unigramas + bigramas `(1,2)`:** considera palabras individuales y pares consecutivos: `no recomiendo`, `recomiendo hotel`.

Los bigramas pueden capturar expresiones como `no recomiendo` o `muy bueno`, aunque suelen aumentar el tamaño del vocabulario. La tabla compara ambas opciones con el resto de condiciones iguales.

### 3.6 TF-IDF

TF-IDF convierte el texto en números que un modelo puede utilizar. Da más peso a términos importantes para un documento y menos a términos que aparecen en muchos documentos.

Parámetros usados en los experimentos:

- `stop_words=sorted(STOPWORDS_ES)`: elimina las stopwords definidas, preservando negaciones importantes.
- `ngram_range=(1,1)` o `(1,2)`: se cambia para comparar unigramas y unigramas+bigrams.
- `min_df=2`: ignora términos que aparecen en menos de dos documentos de entrenamiento.
- `sublinear_tf=True`: aplica una escala logarítmica a la frecuencia del término.
- El vocabulario y los pesos IDF se aprenden **solo con train**; el test no se usa para ajustar el vocabulario.

## 4. Cómo leer la comparación

En `comparacion_modelos_nlp.csv`:

- `cv_f1_macro_media`: métrica principal para comparar configuraciones; promedio de F1 macro en validación cruzada usando solo entrenamiento.
- `cv_f1_macro_std`: variación entre particiones; cuanto menor, más estable puede ser la configuración.
- `cv_accuracy_media`: exactitud media en validación cruzada.
- `test_f1_macro` y `test_accuracy`: métricas en el conjunto reservado de prueba. Sirven como evaluación final, no para escoger repetidamente configuraciones.
- `vocabulario`: cantidad de características aprendidas por TF-IDF.
- `estado`: debe decir `OK` en las seis filas.

F1 macro calcula el F1 de cada clase y promedia dando el mismo peso a cada una; por eso resulta útil si las clases no están equilibradas. No se debe afirmar de antemano que una técnica es la mejor: la conclusión se redacta después de ver los resultados reales.

## 5. Qué comprueba el verificador

`python verificar_experimentos_nlp.py` verifica que existan los archivos de resultados, estén las seis combinaciones y tengan métricas válidas. Si indica que falta una configuración o que la lematización está pendiente, revisa la instalación del modelo español y vuelve a ejecutar `python experimentos_nlp.py`.

## 6. Alcance y límites

El clasificador de referencia (Regresión Logística) se utiliza únicamente para comparar las representaciones NLP de forma consistente. No reemplaza el trabajo de la persona responsable de Modelado, que debe comparar y ajustar modelos según el proyecto.