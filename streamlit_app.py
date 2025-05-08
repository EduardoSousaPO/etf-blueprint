import streamlit as st
import os
import sys
import traceback
from pathlib import Path

# Configurar log mais detalhado
print("=== INICIANDO APLICAÇÃO ETF BLUEPRINT ===")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

# Definir variável de ambiente para Streamlit Cloud
if os.environ.get("STREAMLIT_SHARING_MODE") == "streamlit-sharing" or os.environ.get("IS_STREAMLIT_CLOUD") == "true":
    os.environ["IS_STREAMLIT_CLOUD"] = "true"
    print("Detectado ambiente Streamlit Cloud")

# Garantir que o diretório atual esteja no sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
print(f"Path configurado: {sys.path}")

# Lista arquivos disponíveis para debug
print("Arquivos disponíveis:")
for file in os.listdir():
    print(f" - {file}")

try:
    print("Tentando importar e executar o app principal...")
    import app
    print("App importado com sucesso!")
except Exception as e:
    print(f"ERRO AO INICIAR APP: {str(e)}")
    print(traceback.format_exc())
    
    # Mostrar erro na interface
    st.error("Erro ao iniciar a aplicação")
    st.error(str(e))
    st.code(traceback.format_exc())
    
    # Adicionar informações de debug na interface
    st.subheader("Informações de Debug")
    st.write(f"Python version: {sys.version}")
    st.write(f"Diretório atual: {os.getcwd()}")
    st.write(f"Arquivos disponíveis:")
    for file in os.listdir():
        st.write(f" - {file}") 