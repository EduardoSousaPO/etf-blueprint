#!/bin/bash

echo "========================================"
echo "Configurando ambiente ETF Blueprint"
echo "========================================"

# Instalando pacotes necessários do sistema
echo "Instalando dependências do sistema..."
sudo apt-get update
sudo apt-get install -y python3-dev python3-pip python3-venv build-essential
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.9 python3.9-venv python3.9-dev

# Criando ambiente virtual com Python 3.9
echo "Criando ambiente virtual com Python 3.9..."
python3.9 -m venv venv_etf
source venv_etf/bin/activate

# Atualizando pip
echo "Atualizando pip..."
pip install --upgrade pip setuptools wheel

# Instalando dependências com pip do ambiente virtual
echo "Instalando dependências do projeto..."
pip install --prefer-binary streamlit==1.28.0
pip install --prefer-binary numpy==1.23.5
pip install --prefer-binary pandas==1.5.3
pip install --prefer-binary matplotlib==3.7.2
pip install --prefer-binary plotly==5.18.0
pip install --prefer-binary scipy==1.10.1
pip install --prefer-binary python-dotenv==1.0.0
pip install --prefer-binary openai==1.3.7
pip install --prefer-binary reportlab==4.0.9
pip install --prefer-binary requests==2.31.0
pip install --prefer-binary pydantic==2.4.2
pip install --prefer-binary aiohttp==3.8.6
pip install --prefer-binary python-dateutil==2.8.2

# Executando a aplicação
echo "Iniciando a aplicação Streamlit..."
python -m streamlit run app.py

echo "========================================"
echo "Aplicação encerrada"
echo "========================================" 