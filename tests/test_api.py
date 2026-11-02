"""Pruebas de la API. Ejecutar desde la raíz del repositorio:

    python -m pytest tests
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from api import main  # noqa: E402


@pytest.fixture()
def cliente():
    with TestClient(main.app) as c:
        yield c


def test_salud(cliente):
    r = cliente.get("/api/salud")
    assert r.status_code == 200 and r.json()["estado"] == "ok"


@pytest.mark.parametrize("texto, esperado", [
    ("Habitación sucia y personal grosero, no volvería nunca", "negativo"),
    ("Hotel excelente, desayuno riquísimo y muy buena ubicación", "positivo"),
])
def test_predecir(cliente, texto, esperado):
    r = cliente.post("/api/predecir", json={"texto": texto})
    assert r.status_code == 200
    datos = r.json()
    assert datos["sentimiento"] == esperado
    assert set(datos["probabilidades"]) == {"negativo", "neutro", "positivo"}
    assert abs(sum(datos["probabilidades"].values()) - 1) < 1e-3
    assert datos["confianza"] == datos["probabilidades"][esperado]


def test_predecir_avisa_si_no_hay_palabras_conocidas(cliente):
    r = cliente.post("/api/predecir", json={"texto": "jajjaja"})
    assert r.status_code == 200
    datos = r.json()
    assert datos["sentimiento"] is None and datos["probabilidades"] == {}
    assert datos["aviso"]


def test_predecir_igual_que_el_pipeline(cliente):
    texto = "El hotel está bien, aunque el desayuno era normalito"
    esperado = main._modelo.predict_proba([texto])[0]
    obtenido = cliente.post("/api/predecir", json={"texto": texto}).json()["probabilidades"]
    assert [obtenido[c] for c in main._modelo.classes_] == pytest.approx(esperado, abs=1e-4)


@pytest.mark.parametrize("cuerpo, mensaje", [
    ({"texto": ""}, "La reseña está vacía."),
    ({"texto": "  123 !!! "}, "Escribe una reseña con palabras."),
    ({"texto": "a" * 5001}, "La reseña supera los 5000 caracteres."),
    ({}, "campo \"texto\""),
    ({"texto": 123}, "campo \"texto\""),
])
def test_predecir_rechaza_textos_invalidos_con_mensaje_claro(cliente, cuerpo, mensaje):
    r = cliente.post("/api/predecir", json=cuerpo)
    assert r.status_code == 422
    assert mensaje in r.json()["detail"]


def test_json_mal_formado(cliente):
    r = cliente.post("/api/predecir", content="{texto:", headers={"Content-Type": "application/json"})
    assert r.status_code == 422 and "JSON" in r.json()["detail"]


def test_preprocesar_muestra_cada_paso(cliente):
    r = cliente.post("/api/preprocesar", json={"texto": "¡NO recomiendo este hotel! La habitación estaba sucia"})
    assert r.status_code == 200
    p = r.json()
    assert p["normalizado"] == "no recomiendo este hotel la habitacion estaba sucia"
    assert "no" in p["tokens_sin_stopwords"]                # la negación se conserva
    assert {"este", "la"} <= set(p["stopwords_eliminadas"])
    assert p["raices"][:2] == ["no", "recom"]


def test_preprocesamiento_mostrado_es_el_del_modelo(cliente):
    """El texto procesado que ve el usuario es exactamente lo que recibe el TF-IDF del modelo."""
    texto = "Excelente ubicación, trato amable y camas comodísimas"
    mostrado = cliente.post("/api/preprocesar", json={"texto": texto}).json()["texto_procesado"]
    paso_nlp = main._modelo.named_steps["prep"].named_steps["normalizar"]
    assert paso_nlp.transform([texto]) == [mostrado]


def test_documentacion(cliente):
    assert cliente.get("/docs").status_code == 200
    redoc = cliente.get("/redoc")
    assert redoc.status_code == 200 and "redoc@2." in redoc.text   # versión estable fijada, no "next"


def test_metricas(cliente):
    r = cliente.get("/api/metricas")
    assert r.status_code == 200
    assert r.json()["clases"] == ["negativo", "neutro", "positivo"]


def test_sin_modelo_responde_503(cliente, tmp_path):
    main.cargar_modelo(tmp_path / "no_existe.joblib")
    try:
        assert cliente.get("/api/salud").status_code == 503
        assert cliente.post("/api/predecir", json={"texto": "Muy buen hotel"}).status_code == 503
    finally:
        main.cargar_modelo()
