"""Ejecuta como prueba la verificación de la entrega de Datos y NLP."""
from tests import verificar_entrega


def test_entrega_completa_y_consistente():
    verificar_entrega.main()          # termina con SystemExit(1) si algo falla
