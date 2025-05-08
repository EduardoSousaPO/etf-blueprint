#!/usr/bin/env python3
"""
Script para gerar o arquivo etf_blueprint.zip
Este script cria um arquivo ZIP contendo todos os arquivos necessários
para entrega do projeto, excluindo arquivos desnecessários como .git,
__pycache__, etc.
"""

import os
import zipfile
import glob
import sys
from datetime import datetime

# Diretório do projeto
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Nome do arquivo ZIP de saída
OUTPUT_ZIP = "etf_blueprint.zip"

# Padrões de arquivos para excluir
EXCLUDE_PATTERNS = [
    "__pycache__",
    "*.pyc",
    ".git*",
    ".DS_Store",
    "venv",
    ".env",
    ".pytest_cache",
    "*.zip",
    "temp/*",
    ".ruff_cache",
    ".coverage",
    ".hypothesis",
    "htmlcov",
    "node_modules",
]

def should_include(path):
    """Determina se um caminho deve ser incluído no ZIP"""
    for pattern in EXCLUDE_PATTERNS:
        if glob.fnmatch.fnmatch(path, pattern):
            return False
        if any(glob.fnmatch.fnmatch(part, pattern) for part in path.split(os.sep)):
            return False
    return True

def make_zip():
    """Cria o arquivo ZIP do projeto"""
    print(f"Criando {OUTPUT_ZIP}...")
    
    # Remover ZIP anterior se existir
    if os.path.exists(OUTPUT_ZIP):
        os.remove(OUTPUT_ZIP)
    
    # Criar novo arquivo ZIP
    with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(PROJECT_DIR):
            # Filtrar diretórios para evitar percorrer aqueles que não queremos
            dirs[:] = [d for d in dirs if should_include(os.path.join(root, d))]
            
            # Adicionar arquivos ao ZIP
            for file in files:
                filepath = os.path.join(root, file)
                
                # Verificar se o arquivo é o próprio ZIP que estamos criando
                if filepath == os.path.join(PROJECT_DIR, OUTPUT_ZIP):
                    continue
                
                # Verificar se o arquivo deve ser incluído
                if should_include(filepath):
                    # Caminho relativo para manter a estrutura de diretórios
                    relpath = os.path.relpath(filepath, PROJECT_DIR)
                    print(f"Adicionando: {relpath}")
                    zipf.write(filepath, relpath)
    
    # Verificar tamanho do arquivo
    zip_size = os.path.getsize(OUTPUT_ZIP) / (1024 * 1024)  # Tamanho em MB
    print(f"\nArquivo {OUTPUT_ZIP} criado com sucesso!")
    print(f"Tamanho: {zip_size:.2f} MB")
    print(f"Localização: {os.path.abspath(OUTPUT_ZIP)}")

if __name__ == "__main__":
    make_zip() 