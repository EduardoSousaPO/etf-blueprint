@echo off
echo ETF Blueprint - Iniciando aplicacao via WSL
echo.

REM Tornar o script shell executável
wsl chmod +x run_app_wsl.sh

REM Executar o script shell no WSL
wsl ./run_app_wsl.sh

pause 