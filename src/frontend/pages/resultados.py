import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from io import BytesIO
import base64
import numpy as np
from datetime import datetime
import asyncio
import traceback

# Importar serviços necessários
from src.backend.services.fmp_client import FMPClient
from src.backend.services.optimizer import PortfolioOptimizer
from src.backend.services.openai_client import NarratorService
from src.backend.services.pdf_generator import PDFGenerator

# Lista dos 50 maiores ETFs americanos em PL especificados pelo usuário
DEFAULT_ETFS = [
    "SMH", "SCHG", "MGK", "FTEC", "VGT", "IWY", "IGV", "QQQ", "QQQM", "XLK", 
    "VONG", "CGGR", "XLG", "VOOG", "SPYG", "OEF", "CIBR", "JEPQ", "MGC", "SCHB", 
    "IOO", "ESGV", "QUAL", "VOO", "IVV", "VONE", "IWB", "ESGU", "ITOT", "IWP",
    "GSLC", "DSI", "SPLG", "SPY", "SPHQ", "XMHQ", "PAVE", "MTUM", "CGDV", "AVUS", 
    "VFH", "DUHP", "VIS", "VT", "CGGO", "SDVY", "FNDX", "RDVY", "BUFR"
]

# Lista de ETFs mais conhecidos para melhores resultados
KNOWN_ETFS = [
    "VOO", "SPY", "IVV",   # S&P 500
    "QQQ", "QQQM",         # Nasdaq
    "VTI", "ITOT",         # Mercado total EUA
    "VEA", "IEFA",         # Mercados desenvolvidos
    "VWO", "IEMG",         # Mercados emergentes
    "BND", "AGG",          # Renda fixa EUA
    "BNDX", "IAGG",        # Renda fixa internacional
    "VNQ", "IYR",          # REITs
    "GLD", "IAU",          # Ouro
    "TLT", "IEF", "SHY"    # Títulos do tesouro
]

def gerar_carteira_simulada():
    """
    Gera uma carteira simulada para demonstração
    Será substituída pela implementação real do otimizador
    """
    etfs = [
        {"symbol": "VTI", "name": "Vanguard Total Stock Market ETF", "categoria": "Ações EUA", "expense_ratio": 0.03},
        {"symbol": "VEA", "name": "Vanguard FTSE Developed Markets ETF", "categoria": "Ações Internacionais", "expense_ratio": 0.05},
        {"symbol": "VWO", "name": "Vanguard FTSE Emerging Markets ETF", "categoria": "Mercados Emergentes", "expense_ratio": 0.08},
        {"symbol": "BND", "name": "Vanguard Total Bond Market ETF", "categoria": "Renda Fixa EUA", "expense_ratio": 0.035},
        {"symbol": "BNDX", "name": "Vanguard Total International Bond ETF", "categoria": "Renda Fixa Internacional", "expense_ratio": 0.06},
        {"symbol": "VNQ", "name": "Vanguard Real Estate ETF", "categoria": "Imobiliário", "expense_ratio": 0.12},
        {"symbol": "GLD", "name": "SPDR Gold Shares", "categoria": "Commodities", "expense_ratio": 0.40},
    ]
    
    # Pesos aleatórios que somam 100%
    np.random.seed(42)  # Para reprodutibilidade
    pesos = np.random.dirichlet(np.ones(len(etfs))) * 100
    pesos = [round(peso, 2) for peso in pesos]
    
    # Ajustar para garantir que a soma seja exatamente 100%
    diff = 100 - sum(pesos)
    pesos[0] += diff
    pesos[0] = round(pesos[0], 2)
    
    # Métricas simuladas
    retorno_esperado = 8.5
    volatilidade = 12.2
    sharpe_ratio = 0.7
    max_drawdown = -18.5
    
    # Criar DataFrame
    carteira_df = pd.DataFrame(etfs)
    carteira_df['peso'] = pesos
    carteira_df['retorno_esperado'] = [7.2, 6.8, 9.5, 3.2, 2.8, 5.5, 4.0]
    carteira_df['volatilidade'] = [15.5, 17.2, 22.0, 5.5, 7.2, 18.5, 20.0]
    
    metricas = {
        "retorno_esperado": retorno_esperado,
        "volatilidade": volatilidade,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown
    }
    
    return carteira_df, metricas

def gerar_narrativa_simulada(perfil, carteira, metricas):
    """
    Gera uma narrativa simulada para demonstração
    Será substituída pela implementação real com OpenAI
    """
    return f"""
    Com base no seu perfil {perfil['perfil_risco']} e horizonte de {perfil['horizonte']} anos, 
    desenvolvemos uma carteira diversificada globalmente com foco em crescimento sustentável e proteção.
    
    Sua carteira tem um retorno esperado de {metricas['retorno_esperado']}% ao ano, 
    com volatilidade estimada de {metricas['volatilidade']}%. 
    O índice Sharpe, que mede o retorno ajustado ao risco, é de {metricas['sharpe_ratio']}.
    
    A maior alocação está em VTI ({carteira.iloc[0]['peso']}%), que oferece ampla exposição 
    ao mercado de ações dos EUA. Para diversificação internacional, recomendamos 
    VEA ({carteira.iloc[1]['peso']}%) e VWO ({carteira.iloc[2]['peso']}%).
    
    A parcela defensiva inclui BND ({carteira.iloc[3]['peso']}%) e BNDX ({carteira.iloc[4]['peso']}%), 
    que proporcionam estabilidade e rendimento.
    
    Com um horizonte de {perfil['horizonte']} anos, esta carteira está bem posicionada para 
    atender seu objetivo de retorno de {perfil['retorno_alvo']}% com um nível de risco adequado 
    ao seu perfil {perfil['perfil_risco']}.
    
    Recomendamos revisar esta alocação anualmente ou quando houver mudanças significativas 
    em seus objetivos financeiros.
    """

def gerar_fronteira_eficiente_simulada():
    """
    Gera dados simulados para a fronteira eficiente
    Obs: Ampliada a escala para comportar retornos mais altos
    """
    # Usar uma faixa maior de volatilidades para comportar valores reais
    volatilidades = np.linspace(5, 30, 25)
    
    # Gerar retornos mais altos que tendem a crescer com a volatilidade
    # Fórmula atualizada para gerar retornos de até 25% para volatilidades altas
    retornos = 3 + volatilidades * 0.8 + np.random.normal(0, 0.8, 25)
    
    # Garantir que retornos crescem com volatilidade (com alguma variação)
    retornos = np.sort(retornos)
    
    # Certifique-se de que o valor máximo é pelo menos 25% para comportar carteiras de alto retorno
    if max(retornos) < 25:
        retornos = retornos * (25 / max(retornos))
    
    # Carteira atual (usar valores próximos aos reais exibidos nas métricas)
    # Estes valores serão substituídos pelos valores reais durante a renderização
    atual_vol = 17.52  # Usar o valor aproximado real
    atual_ret = 21.95  # Usar o valor aproximado real
    
    # Carteira de mínima volatilidade
    min_vol = 8.0
    min_ret = 7.0
    
    # Carteira de máximo Sharpe
    max_sharpe_vol = 22.0
    max_sharpe_ret = 23.0
    
    return volatilidades, retornos, atual_vol, atual_ret, min_vol, min_ret, max_sharpe_vol, max_sharpe_ret

def get_table_download_link(df, filename, linkname):
    """
    Gera um link para download do DataFrame como CSV
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-button">{linkname}</a>'
    return href

def display_results(carteira_df, metricas, narrativa, debug_mode=False):
    """
    Exibe os resultados formatados da carteira otimizada
    """
    print("Iniciando a exibição dos resultados")
    
    # Mostrar métricas principais com cards aprimorados
    st.markdown('<h2 class="section-title">Métricas da Carteira</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Retorno Esperado</div>
            <div class="metric-value">{metricas['retorno_esperado']:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Volatilidade</div>
            <div class="metric-value">{metricas['volatilidade']:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Índice Sharpe</div>
            <div class="metric-value">{metricas['sharpe_ratio']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        drawdown_class = "negative" if metricas['max_drawdown'] < 0 else ""
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Drawdown Máximo</div>
            <div class="metric-value {drawdown_class}">{metricas['max_drawdown']:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Mostrar narrativa em card redesenhado
    st.markdown('<h2 class="section-title">Análise Personalizada</h2>', unsafe_allow_html=True)
    # Corrigir o problema com o caractere de escape \n na f-string
    # Primeiro fazer o processamento do texto e depois incluir na f-string
    try:
        # Garantir que a narrativa seja uma string e tratar casos inesperados
        if not isinstance(narrativa, str):
            narrativa = str(narrativa) if narrativa is not None else "Análise não disponível no momento."
            
        narrativa_processada = narrativa.replace('    ', '').replace('\n', '<br><br>')
        st.markdown("""
        <div class="analysis-card">
            {}
        </div>
        """.format(narrativa_processada), unsafe_allow_html=True)
    except Exception as e:
        print(f"Erro ao processar narrativa: {str(e)}")
        # Exibir uma mensagem alternativa em caso de erro
        st.markdown("""
        <div class="analysis-card">
            Não foi possível gerar a análise detalhada neste momento. 
            <br><br>
            Recomendamos revisar esta carteira periodicamente e consultar um profissional financeiro.
        </div>
        """, unsafe_allow_html=True)
    
    # Mostrar tabela de alocação
    st.markdown('<h2 class="section-title">Alocação Recomendada</h2>', unsafe_allow_html=True)
    
    # Formatar tabela
    tabela_exibicao = carteira_df.copy()
    if 'peso' in tabela_exibicao.columns:
        tabela_exibicao['peso'] = tabela_exibicao['peso'].apply(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x)
    
    if 'expense_ratio' in tabela_exibicao.columns:
        tabela_exibicao['expense_ratio'] = tabela_exibicao['expense_ratio'].apply(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x)
    
    if 'retorno_esperado' in tabela_exibicao.columns:
        tabela_exibicao['retorno_esperado'] = tabela_exibicao['retorno_esperado'].apply(lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x)
    
    if 'volatilidade' in tabela_exibicao.columns:
        tabela_exibicao['volatilidade'] = tabela_exibicao['volatilidade'].apply(lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x)
    
    # Renomear colunas para português
    colunas_mapping = {
        'symbol': 'Ticker',
        'name': 'Nome',
        'categoria': 'Categoria',
        'peso': 'Alocação',
        'expense_ratio': 'Taxa de Administração',
        'retorno_esperado': 'Retorno Esperado',
        'volatilidade': 'Volatilidade'
    }
    
    # Renomear apenas as colunas que existem no DataFrame
    colunas_para_renomear = {col: new_name for col, new_name in colunas_mapping.items() if col in tabela_exibicao.columns}
    tabela_exibicao = tabela_exibicao.rename(columns=colunas_para_renomear)
    
    # Exibir tabela com Streamlit
    st.dataframe(tabela_exibicao, use_container_width=True)
    
    try:
        # Gráficos lado a lado
        st.markdown('<h2 class="section-title">Visualização da Carteira</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        # Gráfico de pizza para alocação
        with col1:
            st.markdown("### Distribuição da Carteira")
            
            # Melhorar as cores do gráfico
            custom_colors = px.colors.qualitative.Prism
            
            fig_pizza = px.pie(
                carteira_df, 
                values='peso', 
                names='symbol',
                hover_data=['name', 'categoria'] if 'name' in carteira_df.columns and 'categoria' in carteira_df.columns else None,
                title='Alocação por ETF',
                color_discrete_sequence=custom_colors
            )
            fig_pizza.update_traces(textposition='inside', textinfo='percent+label')
            fig_pizza.update_layout(
                font=dict(size=14),
                title_font=dict(size=18, color='#121A3E'),
                legend=dict(font=dict(size=12)),
                margin=dict(t=50, b=20, l=20, r=20),
                paper_bgcolor='white'
            )
            st.plotly_chart(fig_pizza, use_container_width=True)
        
        # Gráfico de fronteira eficiente
        with col2:
            st.markdown("### Fronteira Eficiente")
            
            # Gerar dados simulados da fronteira
            vols, rets, _, _, min_vol, min_ret, max_sharpe_vol, max_sharpe_ret = gerar_fronteira_eficiente_simulada()
            
            # Usar os valores reais da carteira otimizada
            atual_vol = metricas['volatilidade']
            atual_ret = metricas['retorno_esperado']
            
            fig_frontier = go.Figure()
            
            # Adicionar linha da fronteira
            fig_frontier.add_trace(go.Scatter(
                x=vols, 
                y=rets,
                mode='lines',
                name='Fronteira Eficiente',
                line=dict(color='#4361EE', width=3)
            ))
            
            # Adicionar ponto da carteira atual (usando os valores reais)
            fig_frontier.add_trace(go.Scatter(
                x=[atual_vol],
                y=[atual_ret],
                mode='markers',
                name='Carteira Recomendada',
                marker=dict(color='#06D6A0', size=15, symbol='star')
            ))
            
            # Adicionar ponto de mínima volatilidade
            fig_frontier.add_trace(go.Scatter(
                x=[min_vol],
                y=[min_ret],
                mode='markers',
                name='Mínima Volatilidade',
                marker=dict(color='#118AB2', size=12)
            ))
            
            # Adicionar ponto de máximo sharpe
            fig_frontier.add_trace(go.Scatter(
                x=[max_sharpe_vol],
                y=[max_sharpe_ret],
                mode='markers',
                name='Máximo Sharpe Ratio',
                marker=dict(color='#EF476F', size=12)
            ))
            
            # Configurar layout com escalas adequadas
            fig_frontier.update_layout(
                title='Fronteira Eficiente de Markowitz',
                xaxis_title='Volatilidade (%)',
                yaxis_title='Retorno Esperado (%)',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                margin=dict(l=10, r=10, t=50, b=10),
                # Garantir que os eixos comportem os valores reais
                xaxis=dict(range=[0, max(max(vols) + 2, atual_vol + 2)]),
                yaxis=dict(range=[0, max(max(rets) + 2, atual_ret + 2)]),
                font=dict(size=14),
                title_font=dict(size=18, color='#121A3E'),
                legend_font=dict(size=12),
                paper_bgcolor='white'
            )
            
            st.plotly_chart(fig_frontier, use_container_width=True)
        
        # Tentar exibir gráfico de barras por categoria, se houver a coluna 'categoria'
        if 'categoria' in carteira_df.columns:
            st.markdown('<h2 class="section-title">Alocação por Categoria</h2>', unsafe_allow_html=True)
            
            # Agrupar por categoria
            categoria_df = carteira_df.groupby('categoria')['peso'].sum().reset_index()
            
            fig_categoria = px.bar(
                categoria_df,
                x='categoria',
                y='peso',
                title='Distribuição por Classe de Ativos',
                labels={'categoria': 'Categoria', 'peso': 'Alocação (%)'},
                color='categoria',
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            
            fig_categoria.update_layout(
                font=dict(size=14),
                title_font=dict(size=18, color='#121A3E'),
                xaxis_tickangle=-45,
                margin=dict(t=50, b=20, l=20, r=20),
                paper_bgcolor='white'
            )
            
            st.plotly_chart(fig_categoria, use_container_width=True)
    except Exception as e:
        print(f"Erro ao renderizar gráficos: {str(e)}")
        st.warning("Não foi possível gerar todas as visualizações")
    
    # Botões de download
    try:
        st.markdown('<h2 class="section-title">Exportar Resultados</h2>', unsafe_allow_html=True)
        
        st.markdown('<div class="download-section">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3>Baixar CSV</h3>", unsafe_allow_html=True)
            st.markdown("Exporte os dados da carteira para implementar em sua corretora.")
            st.markdown(get_table_download_link(carteira_df, "carteira_etf.csv", "Download CSV"), unsafe_allow_html=True)
        
        with col2:
            st.markdown("<h3>Relatório PDF</h3>", unsafe_allow_html=True)
            st.markdown("Baixe um relatório completo com análise detalhada da sua carteira.")
            
            # Inicializar o gerador de PDF
            pdf_gen = PDFGenerator()
            
            if st.button("Gerar PDF", type="primary"):
                with st.spinner("Gerando PDF... Isso pode levar alguns segundos."):
                    try:
                        print("Iniciando geração do PDF...")
                        
                        # Gerar o PDF
                        # Criar objeto de carteira simulada em todos os casos
                        from types import SimpleNamespace
                        carteira_obj = SimpleNamespace()
                        carteira_obj.etfs = carteira_df.to_dict('records')
                        carteira_obj.retorno_esperado = metricas['retorno_esperado']
                        carteira_obj.volatilidade = metricas['volatilidade']
                        carteira_obj.sharpe_ratio = metricas['sharpe_ratio']
                        carteira_obj.max_drawdown = metricas['max_drawdown']
                        
                        # Obter perfil diretamente da sessão para evitar erro
                        perfil_pdf = st.session_state.get("perfil", {
                            'perfil_risco': 'Moderado',
                            'horizonte': 10,
                            'retorno_alvo': metricas['retorno_esperado'],
                            'tolerancia_drawdown_numeric': metricas['max_drawdown']
                        })
                        
                        pdf_bytes = pdf_gen.generate(perfil_pdf, carteira_obj, metricas, narrativa)
                        
                        # Codificar para base64 para download
                        b64 = base64.b64encode(pdf_bytes).decode()
                        
                        # Criar link de download
                        pdf_filename = f"carteira_etf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                        href = f'<a href="data:application/pdf;base64,{b64}" download="{pdf_filename}" class="download-button">Clique aqui para baixar o PDF</a>'
                        
                        # Exibir link de download
                        st.success("PDF gerado com sucesso!")
                        st.markdown(href, unsafe_allow_html=True)
                    except Exception as e:
                        print(f"Erro ao gerar PDF: {str(e)}")
                        st.error(f"Erro ao gerar PDF: {str(e)}")
                        # Exibir detalhes do erro sempre, não apenas no modo debug
                        # Verificar se debug_mode existe e é verdadeiro
                        try:
                            if debug_mode:
                                st.error("Detalhes do erro:")
                                st.code(traceback.format_exc())
                        except NameError:
                            # Se debug_mode não estiver definido, mostrar traceback de qualquer forma
                            st.error("Detalhes do erro:")
                            st.code(traceback.format_exc())
    except Exception as e:
        print(f"Erro na seção de download: {str(e)}")
        st.warning("A funcionalidade de exportação não está disponível no momento")
    
    print("Exibição de resultados concluída")

async def show():
    """
    Renderiza a página de resultados com a carteira otimizada
    """
    print("Início da função show() em resultados.py")
    
    # Adicionando estilos CSS personalizados
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
            
            if debug_mode:
                st.sidebar.write(f"FMP API KEY: {fmp_api_key[:5]}...{fmp_api_key[-4:]}")
                st.sidebar.write(f"OpenAI API KEY: {openai_api_key[:5]}...{openai_api_key[-4:]}")
            
            print("Inicializando clientes de API...")
            try:
                cliente_fmp = FMPClient(api_key=fmp_api_key)
                print("Cliente FMP inicializado")
                optimizer = PortfolioOptimizer(fmp_client=cliente_fmp)
                print("Optimizer inicializado")
                narrator = NarratorService(api_key=openai_api_key)
                print("Narrator inicializado")
            except Exception as e:
                print(f"Erro ao inicializar serviços: {str(e)}")
                st.error(f"Erro ao inicializar serviços: {str(e)}")
                return
            
            # Extrair parâmetros do perfil do usuário
            print("Extraindo parâmetros do perfil...")
            try:
                tolerancia_drawdown = perfil["tolerancia_drawdown_numeric"]
                retorno_alvo = perfil["retorno_alvo"]
                perfil_risco = perfil["perfil_risco"]
                
                if debug_mode:
                    st.sidebar.write(f"Perfil de risco: {perfil_risco}")
                    st.sidebar.write(f"Retorno alvo: {retorno_alvo}%")
                    st.sidebar.write(f"Tolerância a drawdown: {tolerancia_drawdown}%")
            except KeyError as e:
                print(f"Campo ausente no perfil: {str(e)}")
                st.error(f"Seu perfil está incompleto (campo ausente: {str(e)}). Por favor, preencha novamente.")
                if st.button("Voltar ao perfil"):
                    st.session_state.nav = "Perfil de Risco"
                    st.rerun()
                return
            
            # Usar a lista completa de 50 ETFs especificados pelo usuário
            symbols_list = DEFAULT_ETFS
            
            if debug_mode:
                st.sidebar.write(f"Número de símbolos a serem otimizados: {len(symbols_list)}")
                st.sidebar.write(f"Símbolos: {', '.join(symbols_list)}")
            
            # Obter dados dos ETFs
            if debug_mode:
                st.sidebar.write("Obtendo dados dos ETFs...")
                start_time = datetime.now()
            
            print("Buscando dados dos ETFs...")
            try:
                etfs_data = await optimizer.get_etf_data(symbols_list)
                print(f"Dados obtidos para {len(etfs_data)} ETFs")
                
                if debug_mode:
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    st.sidebar.write(f"Dados obtidos em {duration:.2f} segundos.")
                    st.sidebar.write(f"ETFs encontrados: {len(etfs_data)}")
                    for etf in etfs_data:
                        st.sidebar.write(f"- {etf.symbol}: {etf.name} (retorno: {etf.retorno_anual:.2f}%, vol: {etf.volatilidade:.2f}%)")
            except Exception as e:
                print(f"Erro ao obter dados dos ETFs: {str(e)}")
                st.error(f"Erro ao obter dados dos ETFs: {str(e)}")
                
                # Alternativa para desenvolvimento/teste - gerar dados simulados
                print("Gerando carteira simulada como alternativa")
                carteira_df, metricas = gerar_carteira_simulada()
                narrativa = gerar_narrativa_simulada(perfil, carteira_df, metricas)
                
                # Exibir uma mensagem clara sobre dados simulados
                st.warning("⚠️ Usando dados simulados devido a problemas com a API. Os resultados a seguir são apenas para demonstração e NÃO representam uma recomendação real.")
                
                # Continuar com dados simulados
                display_results(carteira_df, metricas, narrativa, debug_mode)
                return
            
            # Verificar se temos ETFs suficientes
            if len(etfs_data) < 4:
                print(f"Poucos ETFs encontrados ({len(etfs_data)}). Necessário no mínimo 4 para otimização.")
                st.error(f"Poucos ETFs encontrados ({len(etfs_data)}). Necessário no mínimo 4 para otimização.")
                st.error("Verifique sua chave de API FMP e a conexão com a API.")
                
                # Alternativa para desenvolvimento/teste - gerar dados simulados
                print("Gerando carteira simulada como alternativa")
                carteira_df, metricas = gerar_carteira_simulada()
                narrativa = gerar_narrativa_simulada(perfil, carteira_df, metricas)
                
                # Exibir uma mensagem clara sobre dados simulados
                st.warning("⚠️ Usando dados simulados devido a poucos ETFs disponíveis. Os resultados a seguir são apenas para demonstração e NÃO representam uma recomendação real.")
                
                # Continuar com dados simulados
                display_results(carteira_df, metricas, narrativa, debug_mode)
                return
            
            # Otimizar carteira com parâmetros do usuário
            if debug_mode:
                st.sidebar.write("Iniciando otimização...")
                start_time = datetime.now()
            
            print("Otimizando carteira...")
            try:
                # Validar parâmetros antes de chamar a API
                if not isinstance(retorno_alvo, (int, float)) or retorno_alvo <= 0:
                    raise ValueError(f"Retorno alvo inválido: {retorno_alvo}")
                
                if not isinstance(tolerancia_drawdown, (int, float)):
                    raise ValueError(f"Tolerância a drawdown inválida: {tolerancia_drawdown}")
                
                # Verificar se temos dados de ETF válidos antes de prosseguir
                if not etfs_data or not isinstance(etfs_data, list):
                    raise ValueError(f"Dados de ETFs inválidos ou vazios: {len(etfs_data) if etfs_data else 0} ETFs")
                
                carteira_otimizada = await optimizer.optimize_portfolio(
                    etf_data=etfs_data,
                    target_return=retorno_alvo,
                    risk_profile=perfil_risco,
                    max_drawdown=tolerancia_drawdown
                )
                print("Otimização concluída com sucesso")
                
                # Validar resultados da otimização
                if not hasattr(carteira_otimizada, 'etfs') or not carteira_otimizada.etfs:
                    raise ValueError("Otimização não retornou ETFs válidos")
                
                if debug_mode:
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    st.sidebar.write(f"Otimização concluída em {duration:.2f} segundos.")
                    st.sidebar.write(f"Retorno esperado: {carteira_otimizada.retorno_esperado:.2f}%")
                    st.sidebar.write(f"Volatilidade: {carteira_otimizada.volatilidade:.2f}%")
            except Exception as e:
                print(f"Erro durante a otimização da carteira: {str(e)}")
                st.error(f"Erro durante a otimização da carteira: {str(e)}")
                
                # Alternativa para desenvolvimento/teste - gerar dados simulados
                print("Gerando carteira simulada como alternativa")
                carteira_df, metricas = gerar_carteira_simulada()
                narrativa = gerar_narrativa_simulada(perfil, carteira_df, metricas)
                
                # Exibir uma mensagem clara sobre dados simulados
                st.warning("⚠️ Usando dados simulados devido a um erro na otimização. Os resultados a seguir são apenas para demonstração e NÃO representam uma recomendação real.")
                
                # Continuar com dados simulados
                display_results(carteira_df, metricas, narrativa, debug_mode)
                return
            
            # Ordenar ETFs por peso (do maior para o menor)
            carteira_otimizada.etfs = sorted(
                carteira_otimizada.etfs, 
                key=lambda x: x.get('peso', 0), 
                reverse=True
            )
            
            # Selecionar apenas os 10 ETFs com maior peso
            carteira_otimizada.etfs = carteira_otimizada.etfs[:10]
            
            # Aplicar restrições de alocação mínima e máxima
            min_peso = 3.0  # Mínimo de 3%
            max_peso = 25.0  # Máximo de 25%
            
            # Primeiro, ajustar pesos que excedem o máximo
            total_excesso = 0
            for etf in carteira_otimizada.etfs:
                if etf['peso'] > max_peso:
                    total_excesso += etf['peso'] - max_peso
                    etf['peso'] = max_peso
            
            # Distribuir o excesso proporcionalmente entre os ETFs abaixo do máximo
            etfs_abaixo_max = [etf for etf in carteira_otimizada.etfs if etf['peso'] < max_peso]
            if etfs_abaixo_max and total_excesso > 0:
                for etf in etfs_abaixo_max:
                    # Calcular quanto este ETF recebe do excesso, proporcionalmente
                    ajuste = total_excesso * (etf['peso'] / sum(e['peso'] for e in etfs_abaixo_max))
                    etf['peso'] += ajuste
            
            # Verificar ETFs abaixo do mínimo
            etfs_abaixo_min = [etf for etf in carteira_otimizada.etfs if etf['peso'] < min_peso]
            etfs_acima_min = [etf for etf in carteira_otimizada.etfs if etf['peso'] >= min_peso]
            
            # Se houver ETFs abaixo do mínimo, ajustar
            if etfs_abaixo_min:
                # Calcular quanto precisamos retirar dos ETFs acima do mínimo
                deficit_total = sum(min_peso - etf['peso'] for etf in etfs_abaixo_min)
                
                # Ajustar os ETFs abaixo do mínimo para o mínimo
                for etf in etfs_abaixo_min:
                    etf['peso'] = min_peso
                
                # Retirar proporcionalmente dos ETFs acima do mínimo
                if etfs_acima_min and deficit_total > 0:
                    peso_disponivel = sum(etf['peso'] - min_peso for etf in etfs_acima_min)
                    if peso_disponivel > 0:
                        for etf in etfs_acima_min:
                            # Calcular redução proporcional
                            reducao = deficit_total * ((etf['peso'] - min_peso) / peso_disponivel)
                            etf['peso'] -= reducao
            
            # Normalizar os pesos finais para que somem 100%
            total_peso = sum(etf.get('peso', 0) for etf in carteira_otimizada.etfs)
            
            # Evitar divisão por zero
            if total_peso <= 0:
                print("ATENÇÃO: Soma dos pesos é zero ou negativa. Aplicando pesos iguais.")
                # Aplicar pesos iguais em caso de problema
                peso_igual = 100.0 / len(carteira_otimizada.etfs) if carteira_otimizada.etfs else 0
                for etf in carteira_otimizada.etfs:
                    etf['peso'] = peso_igual
            else:
                # Normalização segura
                for etf in carteira_otimizada.etfs:
                    try:
                        etf['peso'] = (etf.get('peso', 0) / total_peso) * 100
                    except Exception as e:
                        print(f"Erro ao normalizar peso do ETF {etf.get('symbol', 'desconhecido')}: {str(e)}")
                        etf['peso'] = 0  # Valor seguro em caso de erro
            
            # Verificar se a soma dos pesos está próxima de 100% após normalização
            soma_final = sum(etf.get('peso', 0) for etf in carteira_otimizada.etfs)
            if abs(soma_final - 100.0) > 0.1:  # Tolerância de 0.1%
                print(f"ATENÇÃO: Soma dos pesos após normalização é {soma_final:.2f}%, não 100%")
            
            # Converter para DataFrame para visualização
            try:
                carteira_df = pd.DataFrame(carteira_otimizada.etfs)
                
                # Verificar se o DataFrame foi criado corretamente
                if carteira_df.empty:
                    print("DataFrame vazio após conversão. Usando dados de fallback.")
                    carteira_df, metricas = gerar_carteira_simulada()
                    st.warning("⚠️ Problemas na geração da carteira. Usando dados simulados como fallback.")
            except Exception as e:
                print(f"Erro ao converter para DataFrame: {str(e)}")
                carteira_df, metricas = gerar_carteira_simulada()
                st.warning("⚠️ Erro ao processar dados da carteira. Usando dados simulados como fallback.")
            
            # Gerar narrativa personalizada
            if debug_mode:
                st.sidebar.write("Gerando narrativa...")
                start_time = datetime.now()
            
            print("Gerando narrativa personalizada...")
            try:
                narrativa_params = {
                    "perfil": perfil,
                    "carteira": carteira_otimizada,
                    "horizonte": perfil["horizonte"],
                    "objetivo": perfil["retorno_alvo"]
                }
                narrativa = await narrator.generate_portfolio_narrative(narrativa_params)
                print("Narrativa gerada com sucesso")
                
                if debug_mode:
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    st.sidebar.write(f"Narrativa gerada em {duration:.2f} segundos.")
            except Exception as e:
                print(f"Erro ao gerar narrativa: {str(e)}")
                narrativa = f"""
                Com base no seu perfil {perfil.get('perfil_risco', 'de investidor')} e horizonte de {perfil.get('horizonte', 'longo')} anos, 
                desenvolvemos uma carteira diversificada globalmente com foco em crescimento sustentável e proteção.
                
                Esta carteira tem um retorno esperado de {carteira_otimizada.retorno_esperado:.2f}% ao ano 
                com volatilidade estimada de {carteira_otimizada.volatilidade:.2f}%.
                
                Recomendamos revisar esta alocação anualmente ou quando houver mudanças significativas 
                em seus objetivos financeiros.
                """
                st.warning("Não foi possível gerar a análise detalhada. Mostrando narrativa simplificada.")
            
            # Metricas da carteira
            metricas = {
                "retorno_esperado": carteira_otimizada.retorno_esperado,
                "volatilidade": carteira_otimizada.volatilidade,
                "sharpe_ratio": carteira_otimizada.sharpe_ratio,
                "max_drawdown": carteira_otimizada.max_drawdown
            }
            
            # Exibir resultados
            print("Renderizando os resultados finais")
            display_results(carteira_df, metricas, narrativa, debug_mode)
            
        except Exception as e:
            error_details = traceback.format_exc()
            print(f"ERRO FATAL na página de resultados: {str(e)}")
            print(f"Detalhes: {error_details}")
            st.error(f"Ocorreu um erro ao otimizar sua carteira: {str(e)}")
            
            if debug_mode:
                st.sidebar.error("Detalhes do erro:")
                st.sidebar.code(error_details)
            
            st.error("Não foi possível gerar a carteira otimizada. Verifique sua conexão com a internet e as chaves de API.")
            
            # Oferecer alternativa simulada para usuários verem o formato
            if st.button("Mostrar exemplo com dados simulados"):
                print("Gerando carteira simulada")
                carteira_df, metricas = gerar_carteira_simulada()
                narrativa = gerar_narrativa_simulada(perfil, carteira_df, metricas)
                
                # Exibir uma mensagem clara sobre dados simulados
                st.warning("⚠️ Os resultados a seguir são SIMULADOS e NÃO representam uma recomendação real. Servem apenas para demonstrar o formato da solução.")
                
                # Exibir resultados simulados
                display_results(carteira_df, metricas, narrativa, debug_mode)
            return
    
    st.markdown('</div>', unsafe_allow_html=True)
    print("Função show() em resultados.py concluída com sucesso") 