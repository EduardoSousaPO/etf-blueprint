#!/bin/bash

echo "========================================"
echo "ETF Blueprint - Iniciando aplicação no WSL"
echo "========================================"

# Verificar se Python 3.9 está instalado
if ! command -v python3.9 &> /dev/null; then
    echo "Python 3.9 não encontrado. Instalando..."
    sudo apt-get update
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install -y python3.9 python3.9-venv python3.9-dev
fi

# Verificar se o ambiente virtual existe
if [ ! -d "venv39" ]; then
    echo "Criando ambiente virtual com Python 3.9..."
    python3.9 -m venv venv39
fi

# Ativar ambiente virtual
echo "Ativando ambiente virtual..."
source venv39/bin/activate

# Verificar se dependências estão instaladas
if [ ! -f "venv39/.dependencies_installed" ]; then
    echo "Instalando dependências do sistema..."
    sudo apt-get install -y build-essential python3-dev

    echo "Instalando dependências Python..."
    pip install --upgrade pip wheel setuptools
    pip install -r requirements.txt
    
    # Marcar que as dependências foram instaladas
    touch venv39/.dependencies_installed
fi

# Executar a aplicação
echo "Iniciando aplicação Streamlit..."
streamlit run streamlit_app.py

echo "========================================"
echo "Aplicação encerrada"
echo "========================================" 