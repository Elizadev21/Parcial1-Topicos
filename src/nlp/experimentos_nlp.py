"""Experimentos reproducibles de Datos y NLP.

Compara: texto normalizado, stemming Snowball español y lematización spaCy;
TF-IDF unigramas frente a unigramas+bigrams; baseline de clasificación LogisticRegression.
La validación cruzada usa SOLO train.csv. test.csv se evalúa una vez por configuración.

Uso:
  python -m src.nlp.experimentos_nlp
Para habilitar lematización instalar el modelo español:
  python -m spacy download es_core_news_sm
"""
from pathlib import Path
import re, json, warnings
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from nltk.stem.snowball import SnowballStemmer
import spacy
from src.nlp.preprocesamiento import normalizar, STOPWORDS_ES

BASE = Path(__file__).resolve().parents[2]  # raíz del proyecto
DATA = BASE / 'data'
OUT = BASE / 'resultados_nlp'
SEED = 42

class TransformadorMorfologico(BaseEstimator, TransformerMixin):
    """Aplica normalización, opcionalmente stemming o lematización."""
    def __init__(self, metodo='ninguno'):
        self.metodo = metodo

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        textos = [normalizar(x) for x in X]
        if self.metodo == 'ninguno':
            return textos
        if self.metodo == 'stemming':
            stemmer = SnowballStemmer('spanish')
            return [' '.join(stemmer.stem(t) for t in x.split()) for x in textos]
        if self.metodo == 'lematizacion':
            try:
                nlp = spacy.load('es_core_news_sm', disable=['parser', 'ner'])
            except OSError as e:
                raise RuntimeError("Falta el modelo español de spaCy. Ejecuta: python -m spacy download es_core_news_sm") from e
            salida=[]
            for doc in nlp.pipe(textos, batch_size=64):
                salida.append(' '.join(tok.lemma_.lower() for tok in doc if tok.is_alpha))
            return salida
        raise ValueError(f'Método desconocido: {self.metodo}')

def pipeline(metodo, ngram):
    return Pipeline([
        ('morfologia', TransformadorMorfologico(metodo)),
        ('tfidf', TfidfVectorizer(stop_words=sorted(STOPWORDS_ES), ngram_range=ngram,
                                  min_df=2, sublinear_tf=True)),
        ('clasificador', LogisticRegression(max_iter=1500, class_weight='balanced', random_state=SEED))
    ])

def main():
    OUT.mkdir(exist_ok=True)
    train_path, test_path = DATA/'train.csv', DATA/'test.csv'
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError('No encuentro data/train.csv y data/test.csv. Ejecuta primero: python -m src.nlp.preparar_datos')
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    required={'texto','sentimiento'}
    if not required.issubset(train.columns) or not required.issubset(test.columns):
        raise ValueError('Los CSV deben incluir las columnas texto y sentimiento.')
    Xtr, ytr = train.texto.astype(str), train.sentimiento.astype(str)
    Xte, yte = test.texto.astype(str), test.sentimiento.astype(str)
    if set(Xtr).intersection(set(Xte)):
        raise ValueError('Fuga detectada: hay textos repetidos entre train y test.')

    print('=== EJEMPLOS DE PREPROCESAMIENTO ===')
    ejemplos = [
        '¡¡NO recomiendo este hotel!!! Visita https://ejemplo.com en 2024.',
        'La habitación estaba buenoooo, pero no estaba limpia.',
        'Nada limpio y nunca volvería.'
    ]
    filas_ej=[]
    stemmer=SnowballStemmer('spanish')
    for original in ejemplos:
        norm=normalizar(original)
        stem=' '.join(stemmer.stem(t) for t in norm.split())
        filas_ej.append({'original':original,'normalizado':norm,'tokens':str(norm.split()),'stemming':stem,
                         'negacion_no_conservada':('no' in norm.split()),
                         'stopwords_eliminadas':str([t for t in norm.split() if t in STOPWORDS_ES])})
        print('\nOriginal:', original, '\nNormalizado:', norm, '\nTokens:', norm.split(), '\nStemming:', stem)
    pd.DataFrame(filas_ej).to_csv(OUT/'ejemplos_preprocesamiento.csv',index=False,encoding='utf-8-sig')

    metodos=['ninguno','stemming','lematizacion']
    ngrams=[((1,1),'unigramas'),((1,2),'unigramas_bigramas')]
    resultados=[]; detalles=[]
    cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=SEED)
    for metodo in metodos:
        # Detectar falta de modelo antes de ejecutar los 2 experimentos de lematización.
        if metodo=='lematizacion':
            try: spacy.load('es_core_news_sm', disable=['parser','ner'])
            except OSError:
                print('\nAVISO: no está instalado es_core_news_sm; se omite lematización. Instala con: python -m spacy download es_core_news_sm')
                resultados.append({'metodo':metodo,'representacion':'no ejecutado','cv_f1_macro_media':np.nan,
                                   'cv_f1_macro_std':np.nan,'cv_accuracy_media':np.nan,'test_f1_macro':np.nan,
                                   'test_accuracy':np.nan,'vocabulario':np.nan,'estado':'PENDIENTE: instalar modelo spaCy español'})
                continue
        for ng, nombre_ng in ngrams:
            print(f'\n=== {metodo.upper()} + {nombre_ng} ===')
            pipe=pipeline(metodo,ng)
            scores=cross_validate(pipe,Xtr,ytr,cv=cv,scoring={'f1_macro':'f1_macro','accuracy':'accuracy'},
                                  n_jobs=1,return_train_score=False,error_score='raise')
            pipe.fit(Xtr,ytr)
            pred=pipe.predict(Xte)
            vocab=len(pipe.named_steps['tfidf'].vocabulary_)
            f1m=f1_score(yte,pred,average='macro')
            acc=accuracy_score(yte,pred)
            resultados.append({'metodo':metodo,'representacion':nombre_ng,
                'cv_f1_macro_media':scores['test_f1_macro'].mean(),'cv_f1_macro_std':scores['test_f1_macro'].std(),
                'cv_accuracy_media':scores['test_accuracy'].mean(),'test_f1_macro':f1m,'test_accuracy':acc,
                'vocabulario':vocab,'estado':'OK'})
            detalles.append(f'\n\n## {metodo} + {nombre_ng}\n\n'
                f'- F1 macro CV (3 folds, solo train): {scores["test_f1_macro"].mean():.4f} ± {scores["test_f1_macro"].std():.4f}\n'
                f'- Accuracy CV: {scores["test_accuracy"].mean():.4f}\n- F1 macro test: {f1m:.4f}\n- Accuracy test: {acc:.4f}\n- Tamaño del vocabulario: {vocab}\n\n'
                '### Informe por clase (test)\n\n```text\n'+classification_report(yte,pred,zero_division=0)+'\n```\n\n'
                '### Matriz de confusión (filas=real, columnas=predicción; orden de etiquetas alfabético)\n\n```text\n'+str(confusion_matrix(yte,pred))+'\n```')
            print(f'F1 macro CV={scores["test_f1_macro"].mean():.4f}; F1 macro test={f1m:.4f}; accuracy test={acc:.4f}; vocabulario={vocab}')

    dfres=pd.DataFrame(resultados)
    dfres.to_csv(OUT/'comparacion_modelos_nlp.csv',index=False,encoding='utf-8-sig')
    ejecutados=dfres[dfres.estado=='OK']
    mejor=None
    if not ejecutados.empty:
        # Selección basada en CV, nunca en el resultado test.
        mejor=ejecutados.sort_values(['cv_f1_macro_media','cv_accuracy_media'],ascending=False).iloc[0]
    with open(OUT/'informe_comparaciones.md','w',encoding='utf-8') as f:
        f.write('# Informe de experimentos de Datos y NLP\n\n')
        f.write('## Diseño experimental\n\n')
        f.write(f'- Train: {len(train)} filas; test reservado: {len(test)} filas.\n- Semilla: {SEED}.\n- Validación: 3-fold estratificada sobre train solamente.\n')
        f.write('- Clasificador de referencia para hacer comparables las representaciones: LogisticRegression con `class_weight=balanced`. No sustituye la comparación final de modelos del rol Modelado.\n')
        f.write('- TF-IDF: `min_df=2`, `sublinear_tf=True`; comparación `(1,1)` vs `(1,2)`.\n')
        f.write('- Stopwords: se eliminan palabras funcionales, pero se preservan negaciones/intensificadores como `no`, `sin`, `nada`, `nunca`, `muy`, `poco`, porque pueden cambiar el sentimiento.\n')
        f.write('- Test se evalúa después de elegir configuraciones; no se usa para ajustar parámetros. Los resultados de test son diagnóstico final, no criterio de selección.\n')
        f.write('\n## Tabla comparativa\n\n')
        f.write('```text\n'+dfres.to_string(index=False)+'\n```')
        if mejor is not None:
            f.write('\n\n## Mejor configuración según validación cruzada\n\n')
            f.write(f'**{mejor["metodo"]} + {mejor["representacion"]}**, por F1 macro CV medio = {mejor["cv_f1_macro_media"]:.4f}. La selección se basa en CV, no en test.\n')
        f.write('\n\n## Resultados detallados por configuración\n')
        f.write(''.join(detalles))
        f.write('\n\n## Interpretación y límites\n\n')
        f.write('- F1 macro da el mismo peso a cada clase y es importante cuando la distribución de etiquetas está desbalanceada.\n')
        f.write('- Compara primero F1 macro CV y su variabilidad; revisa después las métricas por clase y matriz de confusión.\n')
        f.write('- Stemming recorta palabras y puede crear formas no lingüísticas; lematización intenta llevarlas a su lema y depende del modelo lingüístico instalado.\n')
        f.write('- Este experimento usa un clasificador de referencia para aislar el efecto de las representaciones NLP. El equipo de Modelado debe comparar sus modelos e hiperparámetros con el mismo split y reportar la métrica acordada.\n')
        f.write('- Origen/licencia/fecha de obtención del corpus deben confirmarse en la fuente oficial y documentarse; no se inventan aquí si no están acreditados en los archivos.\n')
    print('\nArchivos generados en:',OUT)
    print('- comparacion_modelos_nlp.csv\n- ejemplos_preprocesamiento.csv\n- informe_comparaciones.md')
    if any(dfres.metodo=='lematizacion') and not (dfres[dfres.metodo=='lematizacion'].estado=='OK').all():
        print('\nPara completar la comparación de lematización instala: python -m spacy download es_core_news_sm y vuelve a ejecutar el script.')

if __name__=='__main__':
    main()
