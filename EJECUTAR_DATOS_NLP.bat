@echo off
setlocal
cd /d "%~dp0"
echo === 1. Preparar datos ===
python -m src.nlp.preparar_datos
if errorlevel 1 goto error
echo === 2. Comparar representaciones NLP ===
python -m src.nlp.experimentos_nlp
if errorlevel 1 goto error
echo === 3. Generar demostraciones paso a paso ===
python -m src.nlp.demostracion_tecnicas_nlp
if errorlevel 1 goto error
echo === 4. Verificar resultados ===
python -m tests.verificar_experimentos_nlp
if errorlevel 1 goto error
echo Proceso terminado. Revisa la carpeta resultados_nlp.
pause
exit /b 0
:error
echo Hubo un error. Lee el mensaje anterior y revisa docs\GUIA_TECNICAS_NLP_PASO_A_PASO.md
pause
exit /b 1
