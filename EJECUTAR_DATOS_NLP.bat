@echo off
setlocal
cd /d "%~dp0"
echo === 1. Preparar datos ===
python preparar_datos.py
if errorlevel 1 goto error
echo === 2. Comparar representaciones NLP ===
python experimentos_nlp.py
if errorlevel 1 goto error
echo === 3. Generar demostraciones paso a paso ===
python demostracion_tecnicas_nlp.py
if errorlevel 1 goto error
echo === 4. Verificar resultados ===
python verificar_experimentos_nlp.py
if errorlevel 1 goto error
echo Proceso terminado. Revisa la carpeta resultados_nlp.
pause
exit /b 0
:error
echo Hubo un error. Lee el mensaje anterior y revisa GUIA_TECNICAS_NLP_PASO_A_PASO.md
pause
exit /b 1
