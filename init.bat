@echo off
echo Iniciando la herramienta ProfileMatchAI...

:: Verifica si ya existe un contenedor con ese nombre y lo elimina
docker rm -f profilematchai >nul 2>&1

:: Ejecuta el contenedor en segundo plano
docker run -d -p 8501:8501 --name profilematchai profile_match_ai

:: Espera unos segundos para que arranque Streamlit
timeout /t 5 >nul

:: Abre el navegador en localhost
start http://localhost:8501