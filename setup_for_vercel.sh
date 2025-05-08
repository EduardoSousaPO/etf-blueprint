#!/bin/bash

echo "========================================"
echo "Preparando ETF Blueprint para deploy Vercel"
echo "========================================"

# Verificando se pip está instalado
if ! command -v pip &> /dev/null; then
    echo "pip não encontrado. Instalando..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python get-pip.py
fi

# Instalando dependências para o deployment
echo "Instalando dependências para deployment..."
pip install -r requirements_prod.txt

echo "========================================"
echo "Projeto pronto para deploy!"
echo "Execute os seguintes comandos para deploy na Vercel:"
echo "vercel login"
echo "vercel"
echo "========================================" 