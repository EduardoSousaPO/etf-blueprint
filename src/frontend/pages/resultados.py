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
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{linkname}</a>'
    return href

async def show():
    """
    Renderiza a página de resultados com a carteira otimizada
    """
    st.title("Resultados - Sua Carteira Otimizada")
    
    # Verificar se há perfil na sessão
    if "perfil" not in st.session_state:
        st.warning("Você precisa preencher seu perfil primeiro.")
        if st.button("Ir para perfil"):
            st.session_state.nav = "Perfil de Risco"
            st.rerun()
        return
    
    # Recuperar dados do perfil
    perfil = st.session_state["perfil"]
    
    # Área de depuração (pode ser removida em produção)
    debug_mode = st.sidebar.checkbox("Modo de depuração", value=False)
    
    # Exibir spinner durante o processamento
    with st.spinner("Otimizando sua carteira de ETFs... Pode levar alguns instantes."):
        try:
            # Inicializar serviços
            fmp_api_key = os.getenv("FMP_API_KEY")
            openai_api_key = os.getenv("OPENAI_API_KEY")
            
            if not fmp_api_key or not openai_api_key:
                st.error("Chaves de API não configuradas. Por favor, configure as variáveis de ambiente FMP_API_KEY e OPENAI_API_KEY.")
                return
            
            if debug_mode:
                st.sidebar.write(f"FMP API KEY: {fmp_api_key[:5]}...{fmp_api_key[-4:]}")
                st.sidebar.write(f"OpenAI API KEY: {openai_api_key[:5]}...{openai_api_key[-4:]}")
            
            cliente_fmp = FMPClient(api_key=fmp_api_key)
            optimizer = PortfolioOptimizer(fmp_client=cliente_fmp)
            narrator = NarratorService(api_key=openai_api_key)
            
            # Extrair parâmetros do perfil do usuário
            tolerancia_drawdown = perfil["tolerancia_drawdown_numeric"]
            retorno_alvo = perfil["retorno_alvo"]
            perfil_risco = perfil["perfil_risco"]
            
            if debug_mode:
                st.sidebar.write(f"Perfil de risco: {perfil_risco}")
                st.sidebar.write(f"Retorno alvo: {retorno_alvo}%")
                st.sidebar.write(f"Tolerância a drawdown: {tolerancia_drawdown}%")
            
            # Usar a lista completa de 50 ETFs especificados pelo usuário
            symbols_list = DEFAULT_ETFS
            
            if debug_mode:
                st.sidebar.write(f"Número de símbolos a serem otimizados: {len(symbols_list)}")
                st.sidebar.write(f"Símbolos: {', '.join(symbols_list)}")
            
            # Obter dados dos ETFs
            if debug_mode:
                st.sidebar.write("Obtendo dados dos ETFs...")
                start_time = datetime.now()
            
            etfs_data = await optimizer.get_etf_data(symbols_list)
            
            if debug_mode:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                st.sidebar.write(f"Dados obtidos em {duration:.2f} segundos.")
                st.sidebar.write(f"ETFs encontrados: {len(etfs_data)}")
                for etf in etfs_data:
                    st.sidebar.write(f"- {etf.symbol}: {etf.name} (retorno: {etf.retorno_anual:.2f}%, vol: {etf.volatilidade:.2f}%)")
            
            # Verificar se temos ETFs suficientes
            if len(etfs_data) < 4:
                st.error(f"Poucos ETFs encontrados ({len(etfs_data)}). Necessário no mínimo 4 para otimização.")
                st.error("Verifique sua chave de API FMP e a conexão com a API.")
                return
            
            # Otimizar carteira com parâmetros do usuário
            if debug_mode:
                st.sidebar.write("Iniciando otimização...")
                start_time = datetime.now()
            
            carteira_otimizada = await optimizer.optimize_portfolio(
                etf_data=etfs_data,
                target_return=retorno_alvo,
                risk_profile=perfil_risco,
                max_drawdown=tolerancia_drawdown
            )
            
            if debug_mode:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                st.sidebar.write(f"Otimização concluída em {duration:.2f} segundos.")
                st.sidebar.write(f"Retorno esperado: {carteira_otimizada.retorno_esperado:.2f}%")
                st.sidebar.write(f"Volatilidade: {carteira_otimizada.volatilidade:.2f}%")
            
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
            for etf in carteira_otimizada.etfs:
                etf['peso'] = (etf.get('peso', 0) / total_peso) * 100
            
            # Converter para DataFrame para visualização
            carteira_df = pd.DataFrame(carteira_otimizada.etfs)
            
            # Gerar narrativa personalizada
            if debug_mode:
                st.sidebar.write("Gerando narrativa...")
                start_time = datetime.now()
            
            narrativa_params = {
                "perfil": perfil,
                "carteira": carteira_otimizada,
                "horizonte": perfil["horizonte"],
                "objetivo": perfil["retorno_alvo"]
            }
            narrativa = await narrator.generate_portfolio_narrative(narrativa_params)
            
            if debug_mode:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                st.sidebar.write(f"Narrativa gerada em {duration:.2f} segundos.")
            
            # Metricas da carteira
            metricas = {
                "retorno_esperado": carteira_otimizada.retorno_esperado,
                "volatilidade": carteira_otimizada.volatilidade,
                "sharpe_ratio": carteira_otimizada.sharpe_ratio,
                "max_drawdown": carteira_otimizada.max_drawdown
            }
        except Exception as e:
            error_details = traceback.format_exc()
            st.error(f"Ocorreu um erro ao otimizar sua carteira: {str(e)}")
            
            if debug_mode:
                st.sidebar.error("Detalhes do erro:")
                st.sidebar.code(error_details)
            
            st.error("Não foi possível gerar a carteira otimizada. Verifique sua conexão com a internet e as chaves de API.")
            return
    
    # Mostrar métricas principais
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Retorno Esperado", f"{metricas['retorno_esperado']}%")
    col2.metric("Volatilidade", f"{metricas['volatilidade']}%")
    col3.metric("Índice Sharpe", f"{metricas['sharpe_ratio']}")
    col4.metric("Drawdown Máximo", f"{metricas['max_drawdown']}%")
    
    # Mostrar narrativa
    st.markdown("## Análise Personalizada")
    st.markdown(f"""
    <div style="background-color: #ffffff; padding: 25px; border-radius: 10px; border-left: 8px solid #1e88e5; box-shadow: 0 2px 5px rgba(0,0,0,0.1); color: #333333; font-size: 16px; line-height: 1.6;">
        {narrativa.replace('    ', '').replace('\n', '<br><br>')}
    </div>
    """, unsafe_allow_html=True)
    
    # Mostrar tabela de alocação
    st.markdown("## Alocação Recomendada")
    
    # Formatar tabela
    tabela_exibicao = carteira_df.copy()
    tabela_exibicao['peso'] = tabela_exibicao['peso'].apply(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x)
    tabela_exibicao['expense_ratio'] = tabela_exibicao['expense_ratio'].apply(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x)
    
    if 'retorno_esperado' in tabela_exibicao.columns:
        tabela_exibicao['retorno_esperado'] = tabela_exibicao['retorno_esperado'].apply(lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x)
    
    if 'volatilidade' in tabela_exibicao.columns:
        tabela_exibicao['volatilidade'] = tabela_exibicao['volatilidade'].apply(lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x)
    
    # Renomear colunas para português
    tabela_exibicao = tabela_exibicao.rename(columns={
        'symbol': 'Ticker',
        'name': 'Nome',
        'categoria': 'Categoria',
        'peso': 'Alocação',
        'expense_ratio': 'Taxa de Administração',
        'retorno_esperado': 'Retorno Esperado',
        'volatilidade': 'Volatilidade'
    })
    
    # Exibir tabela com Streamlit
    st.dataframe(tabela_exibicao, use_container_width=True)
    
    # Gráficos lado a lado
    col1, col2 = st.columns(2)
    
    # Gráfico de pizza para alocação
    with col1:
        st.markdown("### Distribuição da Carteira")
        fig_pizza = px.pie(
            carteira_df, 
            values='peso', 
            names='symbol',
            hover_data=['name', 'categoria'],
            title='Alocação por ETF',
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig_pizza.update_traces(textposition='inside', textinfo='percent+label')
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
            line=dict(color='blue', width=2)
        ))
        
        # Adicionar ponto da carteira atual (usando os valores reais)
        fig_frontier.add_trace(go.Scatter(
            x=[atual_vol],
            y=[atual_ret],
            mode='markers',
            name='Carteira Recomendada',
            marker=dict(color='green', size=12, symbol='star')
        ))
        
        # Adicionar ponto de mínima volatilidade
        fig_frontier.add_trace(go.Scatter(
            x=[min_vol],
            y=[min_ret],
            mode='markers',
            name='Mínima Volatilidade',
            marker=dict(color='blue', size=10)
        ))
        
        # Adicionar ponto de máximo sharpe
        fig_frontier.add_trace(go.Scatter(
            x=[max_sharpe_vol],
            y=[max_sharpe_ret],
            mode='markers',
            name='Máximo Sharpe Ratio',
            marker=dict(color='red', size=10)
        ))
        
        # Configurar layout com escalas adequadas
        fig_frontier.update_layout(
            title='Fronteira Eficiente de Markowitz',
            xaxis_title='Volatilidade (%)',
            yaxis_title='Retorno Esperado (%)',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=10, r=10, t=30, b=10),
            # Garantir que os eixos comportem os valores reais
            xaxis=dict(range=[0, max(max(vols) + 2, atual_vol + 2)]),
            yaxis=dict(range=[0, max(max(rets) + 2, atual_ret + 2)])
        )
        
        st.plotly_chart(fig_frontier, use_container_width=True)
    
    # Gráfico de barras por categoria
    st.markdown("### Alocação por Categoria")
    
    # Agrupar por categoria
    categoria_df = carteira_df.groupby('categoria')['peso'].sum().reset_index()
    
    fig_categoria = px.bar(
        categoria_df,
        x='categoria',
        y='peso',
        title='Distribuição por Classe de Ativos',
        labels={'categoria': 'Categoria', 'peso': 'Alocação (%)'},
        color='categoria',
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    
    st.plotly_chart(fig_categoria, use_container_width=True)
    
    # Botões de download
    st.markdown("## Exportar Resultados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Baixar CSV")
        st.markdown("Exporte os dados da carteira para implementar em sua corretora.")
        st.markdown(get_table_download_link(carteira_df, "carteira_etf.csv", "Download CSV"), unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Relatório PDF")
        st.markdown("Baixe um relatório completo com análise detalhada da sua carteira.")
        
        # Inicializar o gerador de PDF
        pdf_gen = PDFGenerator()
        
        if st.button("Gerar PDF", type="primary"):
            with st.spinner("Gerando PDF... Isso pode levar alguns segundos."):
                try:
                    # Log para debug
                    st.write("Iniciando geração do PDF...")
                    
                    # Gerar o PDF
                    pdf_bytes = pdf_gen.generate(perfil, carteira_otimizada, metricas, narrativa)
                    
                    # Codificar para base64 para download
                    b64 = base64.b64encode(pdf_bytes).decode()
                    
                    # Criar link de download
                    pdf_filename = f"carteira_etf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    href = f'<a href="data:application/pdf;base64,{b64}" download="{pdf_filename}">Clique aqui para baixar o PDF</a>'
                    
                    # Exibir link de download
                    st.success("PDF gerado com sucesso!")
                    st.markdown(href, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Erro ao gerar PDF: {str(e)}")
                    # Exibir detalhes do erro sempre, não apenas no modo debug
                    st.error("Detalhes do erro:")
                    st.code(traceback.format_exc()) 