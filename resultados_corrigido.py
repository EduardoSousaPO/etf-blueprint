"""
Este é um fragmento de código para substituir a parte com problemas no arquivo resultados.py.
Substitua a função show() inteira com este código corrigido.
"""
import streamlit as st
import os
from datetime import datetime

async def show():
    """
    Renderiza a página de resultados com a carteira otimizada
    """
    print("Início da função show() em resultados.py")
    
    # Adicionando estilos CSS personalizados usando string normal (não f-string)
    st.markdown("""
    <style>
        /* Estilo geral */
        .results-page {
            padding: 1rem 0;
        }
        
        /* Estilos para os cards de métricas */
        .metric-card {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.06);
            text-align: center;
            height: 100%;
            transition: transform 0.3s;
        }
        .metric-card:hover {
            transform: translateY(-5px);
        }
        .metric-title {
            font-size: 1rem;
            color: #4F5665;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #4361EE;
        }
        .metric-value.negative {
            color: #EF476F;
        }
        
        /* Estilos para a análise personalizada */
        .analysis-card {
            background-color: white;
            border-radius: 10px;
            border-left: 8px solid #4361EE;
            padding: 25px;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.06);
            color: #333333;
            font-size: 16px;
            line-height: 1.6;
            margin-bottom: 2rem;
        }
        
        /* Estilos para a tabela de alocação */
        .stDataFrame {
            border-radius: 10px !important;
            overflow: hidden !important;
        }
        .dataframe {
            border-collapse: collapse !important;
            width: 100% !important;
        }
        .dataframe th {
            background-color: #4361EE !important;
            color: white !important;
            font-weight: 600 !important;
            text-align: left !important;
            padding: 12px !important;
        }
        .dataframe td {
            padding: 12px !important;
            border-bottom: 1px solid #f0f0f0 !important;
        }
        .dataframe tr:hover {
            background-color: #f8f9fe !important;
        }
        
        /* Estilos para os botões de download */
        .download-section {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.06);
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .download-button {
            display: inline-block;
            background-color: #4361EE;
            color: white !important;
            padding: 10px 20px;
            border-radius: 5px;
            text-decoration: none;
            font-weight: 600;
            text-align: center;
            transition: background-color 0.3s;
            margin-top: 10px;
        }
        .download-button:hover {
            background-color: #3A0CA3;
            text-decoration: none;
        }
        
        /* Título da seção com linha destaque */
        .section-title {
            position: relative;
            padding-bottom: 10px;
            margin-bottom: 20px;
            font-weight: 600;
            color: #121A3E;
        }
        .section-title:after {
            content: "";
            position: absolute;
            left: 0;
            bottom: 0;
            width: 50px;
            height: 3px;
            background-color: #4361EE;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="results-page">', unsafe_allow_html=True)
    st.title("Resultados - Sua Carteira Otimizada")
    
    # Verificar se há perfil na sessão
    if "perfil" not in st.session_state:
        print("Perfil não encontrado na sessão. Redirecionando para a página de perfil.")
        st.warning("Você precisa preencher seu perfil primeiro.")
        if st.button("Ir para perfil"):
            st.session_state.nav = "Perfil de Risco"
            st.rerun()
        return
    
    # Recuperar dados do perfil
    try:
        perfil = st.session_state["perfil"]
        print(f"Perfil recuperado: {perfil}")
    except Exception as e:
        print(f"Erro ao recuperar perfil: {str(e)}")
        st.error("Erro ao recuperar seu perfil. Por favor, preencha novamente.")
        if st.button("Ir para perfil"):
            st.session_state.nav = "Perfil de Risco"
            st.rerun()
        return
    
    # Área de depuração (pode ser removida em produção)
    debug_mode = st.sidebar.checkbox("Modo de depuração", value=False)
    
    # Exibir spinner durante o processamento
    with st.spinner("Otimizando sua carteira de ETFs... Pode levar alguns instantes."):
        try:
            # Inicializar serviços
            fmp_api_key = os.getenv("FMP_API_KEY")
            openai_api_key = os.getenv("OPENAI_API_KEY")
            
            if not fmp_api_key or not openai_api_key:
                print("Chaves de API não configuradas")
                st.error("Chaves de API não configuradas. Por favor, configure as variáveis de ambiente FMP_API_KEY e OPENAI_API_KEY.")
                return
                
            # O restante do código da função show seria incluído aqui...
            # Este é apenas um esboço para corrigir o erro de indentação
            st.info("Carregando dados...")
        except Exception as e:
            st.error(f"Ocorreu um erro: {str(e)}")
            return 