@echo off
echo Iniciando ETF Blueprint via WSL...
wsl bash -c "cd /mnt/c/Users/edusp/Projetos_App_Desktop/projeto-ETF-PDF && python3 -m venv venv_etf && ./venv_etf/bin/pip install -r requirements.txt && ./venv_etf/bin/streamlit run app.py"
pause 