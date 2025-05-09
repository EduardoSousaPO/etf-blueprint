import streamlit as st
import sys
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
import traceback

# Carregar variáveis de ambiente do arquivo .env com tratamento de erros
try:
    load_dotenv()
    print("Arquivo .env carregado com sucesso")
except Exception as e:
    print(f"Erro ao carregar arquivo .env: {str(e)}")
    print("Continuando sem o arquivo .env...")

# Verificar se as variáveis foram carregadas
fmp_api_key = os.getenv("FMP_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Verificar se estamos no ambiente Streamlit Cloud
is_streamlit_cloud = os.getenv("IS_STREAMLIT_CLOUD") == "true"

# Desativar o modo de demonstração - usar apenas dados reais
st.session_state.demo_mode = False

# Verificar se as chaves API estão disponíveis
if not fmp_api_key or not openai_api_key:
    st.error("⚠️ Chaves de API necessárias não encontradas. Configure o arquivo .env com FMP_API_KEY e OPENAI_API_KEY.")
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

# Esconder o menu do Streamlit e o rodapé "Made with Streamlit"
hide_menu_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display:none;}
        header {visibility: hidden;}
        
        /* Design responsivo para mobile */
        @media (max-width: 768px) {
            .stApp {
                padding-top: 10px;
            }
            .stButton button {
                width: 100%;
                margin-bottom: 10px;
            }
        }
        
        /* Melhorar a aparência da barra de navegação no sidebar */
        .stRadio > div {
            padding: 10px;
            border-radius: 8px;
            cursor: pointer;
        }
        .stRadio > div:hover {
            background-color: #F8F9FE;
        }
        .stSidebar .stRadio [role=radiogroup] {
            gap: 10px;
        }
        .stSidebar .stRadio [data-testid=stMarkdownContainer] > p {
            font-size: 1.05rem !important;
            font-weight: 600 !important;
            padding-left: 10px;
        }
    </style>
    """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# Inicializa o session_state para navegação, se necessário
if 'nav' not in st.session_state:
    st.session_state.nav = "Home"

# Verifica parâmetros de URL para navegação
try:
    print("Tentando acessar query_params")
    # Método mais seguro para verificar se o recurso está disponível
    query_params_existe = 'query_params' in dir(st)
    print(f"query_params existe? {query_params_existe}")
    
    if query_params_existe:
        try:
            # Verificar tanto "page" quanto "nav" como parâmetros de URL
            page = st.query_params.get("page", "")
            nav_param = st.query_params.get("nav", "")
            
            print(f"Parâmetro page encontrado: {page}")
            print(f"Parâmetro nav encontrado: {nav_param}")
            
            if page == "perfil" or nav_param == "perfil":
                print("Navegando para Perfil de Risco")
                st.session_state.nav = "Perfil de Risco"
            elif page == "resultados" or nav_param == "resultados":
                print("Navegando para Resultados")
                st.session_state.nav = "Resultados"
            # Limpar os parâmetros após uso
            try:
                st.query_params.clear()
            except:
                pass
        except Exception as e:
            print(f"Erro ao processar query_params: {str(e)}")
except Exception as e:
    print(f"ERRO ao acessar query_params: {str(e)}")
    # Se não for possível acessar query_params, continuar com a navegação padrão
    pass

# Sidebar para navegação
st.sidebar.title("ETF Blueprint")
st.sidebar.markdown("""
<div style="margin-bottom: 20px; border-radius: 8px; padding: 10px; background-color: #F8F9FE;">
    <p style="margin: 0; font-size: 0.9rem; color: #4F5665;">
        Otimização de carteiras de ETFs com algoritmos avançados
    </p>
</div>
""", unsafe_allow_html=True)

# Detectar índice atual de forma segura
try:
    current_index = ["Home", "Perfil de Risco", "Resultados"].index(st.session_state.nav)
except:
    current_index = 0
    st.session_state.nav = "Home"

opcao = st.sidebar.radio(
    "Navegação",
    ["Home", "Perfil de Risco", "Resultados"],
    key="sidebar_nav",
    index=current_index
)

# Atualiza o session_state quando a navegação é alterada pelo sidebar
if opcao != st.session_state.nav:
    st.session_state.nav = opcao
    # Usar try/except para maior compatibilidade
    try:
        st.experimental_rerun()
    except:
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
    # Verificar se as chaves API necessárias estão disponíveis antes de mostrar a página de resultados
    if not fmp_api_key or not openai_api_key:
        st.error("⚠️ Chaves de API necessárias não encontradas. Configure o arquivo .env com FMP_API_KEY e OPENAI_API_KEY antes de acessar os resultados.")
        st.info("Volte para a Home e configure as chaves API para continuar.")
    else:
        try:
            run_async(show_resultados) 
        except Exception as e:
            st.error(f"Erro ao carregar a página de resultados: {str(e)}")
            st.error(f"Detalhes: {traceback.format_exc()}")
            st.info("Recarregue a página ou volte para a Home para continuar.") 