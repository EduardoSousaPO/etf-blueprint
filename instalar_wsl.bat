@echo off
echo =================================
echo    Instalacao do WSL para ETF Blueprint
echo =================================
echo.

REM Verificar se o WSL já está instalado
wsl --status > nul 2>&1
if %errorlevel% equ 0 (
    echo WSL ja esta instalado! Voce pode executar a aplicacao usando run_app_wsl.bat
    pause
    exit
)

echo O WSL nao esta instalado ou nao esta ativo.
echo Este script vai instalar o WSL com Ubuntu por padrao.
echo.
echo ATENCAO: Este processo requer privilegios de administrador e pode
echo necessitar de reinicializacao do sistema.
echo.

set /p confirmacao="Deseja prosseguir com a instalacao do WSL? (S/N): "
if /i "%confirmacao%" neq "S" (
    echo Instalacao cancelada.
    pause
    exit
)

echo.
echo Instalando WSL...
echo.

REM Instalar WSL com Ubuntu
powershell -Command "Start-Process PowerShell -Verb RunAs -ArgumentList '-Command', 'wsl --install'"

echo.
echo Se nao houver erros, o WSL com Ubuntu esta sendo instalado.
echo Apos a conclusao e possivel reinicializacao, execute o script run_app_wsl.bat para iniciar a aplicacao.
echo.
pause 