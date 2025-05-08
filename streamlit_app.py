import streamlit as st
import os
import sys
from pathlib import Path

# Definir variável de ambiente para Streamlit Cloud
if os.environ.get("STREAMLIT_SHARING_MODE") == "streamlit-sharing" or os.environ.get("IS_STREAMLIT_CLOUD") == "true":
    os.environ["IS_STREAMLIT_CLOUD"] = "true"
    print("Detectado ambiente Streamlit Cloud")

# Garantir que o diretório atual esteja no sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importando e executando o app principal
import app 