import streamlit as st
import os
import sys
from pathlib import Path

# Garantir que o diretório atual esteja no sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Desabilitando warnings de depreciação
st.set_option('deprecation.showPyplotGlobalUse', False)

# Importando e executando o app principal
import app 