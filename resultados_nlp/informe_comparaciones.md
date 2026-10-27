# Informe de experimentos de Datos y NLP

## Diseño experimental

- Train: 1433 filas; test reservado: 359 filas.
- Semilla: 42.
- Validación: 3-fold estratificada sobre train solamente.
- Clasificador de referencia para hacer comparables las representaciones: LogisticRegression con `class_weight=balanced`. No sustituye la comparación final de modelos del rol Modelado.
- TF-IDF: `min_df=2`, `sublinear_tf=True`; comparación `(1,1)` vs `(1,2)`.
- Stopwords: se eliminan palabras funcionales, pero se preservan negaciones/intensificadores como `no`, `sin`, `nada`, `nunca`, `muy`, `poco`, porque pueden cambiar el sentimiento.
- Test se evalúa después de elegir configuraciones; no se usa para ajustar parámetros. Los resultados de test son diagnóstico final, no criterio de selección.

## Tabla comparativa

```text
      metodo     representacion  cv_f1_macro_media  cv_f1_macro_std  cv_accuracy_media  test_f1_macro  test_accuracy  vocabulario estado
     ninguno          unigramas           0.715360         0.035866           0.792757       0.707571       0.782730         5839     OK
     ninguno unigramas_bigramas           0.716135         0.028727           0.803917       0.685876       0.777159        15485     OK
    stemming          unigramas           0.714733         0.035160           0.791367       0.720219       0.796657         3634     OK
    stemming unigramas_bigramas           0.719858         0.032573           0.810901       0.709106       0.810585        16099     OK
lematizacion          unigramas           0.707476         0.024866           0.789270       0.713620       0.799443         4462     OK
lematizacion unigramas_bigramas           0.709444         0.036375           0.804616       0.720305       0.810585        16608     OK
```

## Mejor configuración según validación cruzada

**stemming + unigramas_bigramas**, por F1 macro CV medio = 0.7199. La selección se basa en CV, no en test.


## Resultados detallados por configuración


## ninguno + unigramas

- F1 macro CV (3 folds, solo train): 0.7154 ± 0.0359
- Accuracy CV: 0.7928
- F1 macro test: 0.7076
- Accuracy test: 0.7827
- Tamaño del vocabulario: 5839

### Informe por clase (test)

```text
              precision    recall  f1-score   support

    negativo       0.80      0.86      0.83       100
      neutro       0.42      0.44      0.43        57
    positivo       0.88      0.84      0.86       202

    accuracy                           0.78       359
   macro avg       0.70      0.71      0.71       359
weighted avg       0.79      0.78      0.78       359

```

### Matriz de confusión (filas=real, columnas=predicción; orden de etiquetas alfabético)

```text
[[ 86   9   5]
 [ 14  25  18]
 [  7  25 170]]
```

## ninguno + unigramas_bigramas

- F1 macro CV (3 folds, solo train): 0.7161 ± 0.0287
- Accuracy CV: 0.8039
- F1 macro test: 0.6859
- Accuracy test: 0.7772
- Tamaño del vocabulario: 15485

### Informe por clase (test)

```text
              precision    recall  f1-score   support

    negativo       0.81      0.85      0.83       100
      neutro       0.38      0.35      0.37        57
    positivo       0.86      0.86      0.86       202

    accuracy                           0.78       359
   macro avg       0.69      0.69      0.69       359
weighted avg       0.77      0.78      0.77       359

```

### Matriz de confusión (filas=real, columnas=predicción; orden de etiquetas alfabético)

```text
[[ 85   9   6]
 [ 15  20  22]
 [  5  23 174]]
```

## stemming + unigramas

- F1 macro CV (3 folds, solo train): 0.7147 ± 0.0352
- Accuracy CV: 0.7914
- F1 macro test: 0.7202
- Accuracy test: 0.7967
- Tamaño del vocabulario: 3634

### Informe por clase (test)

```text
              precision    recall  f1-score   support

    negativo       0.80      0.89      0.84       100
      neutro       0.41      0.46      0.43        57
    positivo       0.92      0.85      0.88       202

    accuracy                           0.80       359
   macro avg       0.71      0.73      0.72       359
weighted avg       0.81      0.80      0.80       359

```

### Matriz de confusión (filas=real, columnas=predicción; orden de etiquetas alfabético)

```text
[[ 89  11   0]
 [ 17  26  14]
 [  5  26 171]]
```

## stemming + unigramas_bigramas

- F1 macro CV (3 folds, solo train): 0.7199 ± 0.0326
- Accuracy CV: 0.8109
- F1 macro test: 0.7091
- Accuracy test: 0.8106
- Tamaño del vocabulario: 16099

### Informe por clase (test)

```text
              precision    recall  f1-score   support

    negativo       0.83      0.89      0.86       100
      neutro       0.42      0.33      0.37        57
    positivo       0.88      0.91      0.89       202

    accuracy                           0.81       359
   macro avg       0.71      0.71      0.71       359
weighted avg       0.80      0.81      0.80       359

```

### Matriz de confusión (filas=real, columnas=predicción; orden de etiquetas alfabético)

```text
[[ 89   9   2]
 [ 16  19  22]
 [  2  17 183]]
```

## lematizacion + unigramas

- F1 macro CV (3 folds, solo train): 0.7075 ± 0.0249
- Accuracy CV: 0.7893
- F1 macro test: 0.7136
- Accuracy test: 0.7994
- Tamaño del vocabulario: 4462

### Informe por clase (test)

```text
              precision    recall  f1-score   support

    negativo       0.83      0.87      0.85       100
      neutro       0.41      0.40      0.41        57
    positivo       0.89      0.88      0.89       202

    accuracy                           0.80       359
   macro avg       0.71      0.72      0.71       359
weighted avg       0.80      0.80      0.80       359

```

### Matriz de confusión (filas=real, columnas=predicción; orden de etiquetas alfabético)

```text
[[ 87  11   2]
 [ 15  23  19]
 [  3  22 177]]
```

## lematizacion + unigramas_bigramas

- F1 macro CV (3 folds, solo train): 0.7094 ± 0.0364
- Accuracy CV: 0.8046
- F1 macro test: 0.7203
- Accuracy test: 0.8106
- Tamaño del vocabulario: 16608

### Informe por clase (test)

```text
              precision    recall  f1-score   support

    negativo       0.82      0.88      0.85       100
      neutro       0.46      0.39      0.42        57
    positivo       0.89      0.90      0.89       202

    accuracy                           0.81       359
   macro avg       0.72      0.72      0.72       359
weighted avg       0.80      0.81      0.81       359

```

### Matriz de confusión (filas=real, columnas=predicción; orden de etiquetas alfabético)

```text
[[ 88   9   3]
 [ 15  22  20]
 [  4  17 181]]
```

## Interpretación y límites

- F1 macro da el mismo peso a cada clase y es importante cuando la distribución de etiquetas está desbalanceada.
- Compara primero F1 macro CV y su variabilidad; revisa después las métricas por clase y matriz de confusión.
- Stemming recorta palabras y puede crear formas no lingüísticas; lematización intenta llevarlas a su lema y depende del modelo lingüístico instalado.
- Este experimento usa un clasificador de referencia para aislar el efecto de las representaciones NLP. El equipo de Modelado debe comparar sus modelos e hiperparámetros con el mismo split y reportar la métrica acordada.
- Origen/licencia/fecha de obtención del corpus deben confirmarse en la fuente oficial y documentarse; no se inventan aquí si no están acreditados en los archivos.
