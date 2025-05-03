import streamlit as st
import sys
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Verificar se as variáveis foram carregadas
fmp_api_key = os.getenv("FMP_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not fmp_api_key or not openai_api_key:
    st.error("Erro: Variáveis de ambiente não encontradas. Verifique o arquivo .env")
    print(f"FMP_API_KEY: {'Encontrada' if fmp_api_key else 'Não encontrada'}")
    print(f"OPENAI_API_KEY: {'Encontrada' if openai_api_key else 'Não encontrada'}")

# Adiciona o diretório src ao path para importação dos módulos
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.frontend.pages.home import show as show_home
from src.frontend.pages.perfil import show as show_perfil
from src.frontend.pages.resultados import show as show_resultados

# Configurações da página
st.set_page_config(
    page_title="ETF Blueprint - Carteiras Personalizadas",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializa o session_state para navegação, se necessário
if 'nav' not in st.session_state:
    st.session_state.nav = "Home"

# Verifica parâmetros de URL para navegação
if "page" in st.query_params:
    page = st.query_params["page"].lower()
    if page == "perfil":
        st.session_state.nav = "Perfil de Risco"
    elif page == "resultados":
        st.session_state.nav = "Resultados"
    # Limpar os parâmetros após uso
    st.query_params.clear()

# Sidebar para navegação
st.sidebar.title("ETF Blueprint")
opcao = st.sidebar.radio(
    "Navegação",
    ["Home", "Perfil de Risco", "Resultados"],
    key="sidebar_nav",
    index=["Home", "Perfil de Risco", "Resultados"].index(st.session_state.nav)
)

# Atualiza o session_state quando a navegação é alterada pelo sidebar
if opcao != st.session_state.nav:
    st.session_state.nav = opcao
    st.rerun()

# Função para executar tarefas assíncronas
def run_async(func):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(func())
    loop.close()
    return result

# Renderiza a página com base no session_state
if st.session_state.nav == "Home":
    show_home()
elif st.session_state.nav == "Perfil de Risco":
    show_perfil()
elif st.session_state.nav == "Resultados":
    run_async(show_resultados) 