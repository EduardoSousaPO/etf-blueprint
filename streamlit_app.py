import streamlit as st
import os

# Configurando o título e ícone da página
st.set_page_config(
    page_title="ETF Blueprint",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Importando e executando o app principal
import app 