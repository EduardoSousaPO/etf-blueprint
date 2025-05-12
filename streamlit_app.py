import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import requests
import json
import openai
from dotenv import load_dotenv
from datetime import datetime
import io
from PIL import Image
import matplotlib.pyplot as plt
import cvxpy as cp

# Configuração Inicial
st.set_page_config(
    page_title="ETF Blueprint",
    page_icon="📈",
    layout="wide"
)

# Esconder o menu do Streamlit e o rodapé
hide_menu_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display:none;}
    </style>
    """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# Carregar variáveis de ambiente
load_dotenv()

# Chaves API
FMP_API_KEY = os.getenv("FMP_API_KEY", st.secrets.get("FMP_API_KEY", ""))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))

# Configurar OpenAI
openai.api_key = OPENAI_API_KEY

# Definir listas de ETFs
ETFS_BR = [
    "BBSD11", "BOVA11", "BOVB11", "BOVS11", "BOVV11", "BRAX11", 
    "DIVO11", "ECOO11", "FIND11", "GOVE11", "HASH11", "ISPT11", 
    "IVVB11", "META11", "PIBB11", "QBTC11", "QETH11", "RICH11", 
    "SMAL11", "URA11", "XFIX11", "XINA11"
]

ETFS_EUA = [
    "SPY", "QQQ", "IWM", "DIA", "VTI", "IEFA", "EFA", "IEMG", "EEM", 
    "AGG", "BND", "LQD", "HYG", "TLT", "IEF", "GLD", "IAU", "SLV", 
    "XLK", "XLF", "XLV", "XLE", "XLY", "XLI", "XLP", "XLU", "XLB", 
    "XLC", "XLRE", "VGT", "VHT", "VOX", "VCR", "VAW", "VPU", "VFH", 
    "VDE", "VIS", "VDC", "VNQ", "IYR", "ARKK", "ARKW", "ARKF", "ARKG", 
    "ARKX", "QYLD", "JEPI", "JEPQ", "SCHD"
]

# Função para filtrar universo de ETFs
def filtrar_universo(opcao):
    if opcao == "BR":
        return ETFS_BR
    if opcao == "EUA":
        return ETFS_EUA
    return ETFS_BR + ETFS_EUA

# Função para obter preços históricos
def get_prices(tickers, years=5):
    # Usar cache do session_state se disponível
    cache_key = f"prices_cache_{','.join(tickers)}_{years}"
    
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    
    all_data = pd.DataFrame()
    
    with st.spinner(f"Obtendo dados históricos para {len(tickers)} ETFs..."):
        for ticker in tickers:
            try:
                url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={FMP_API_KEY}"
                response = requests.get(url)
                data = response.json()
                
                if "historical" in data:
                    df = pd.DataFrame(data["historical"])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date')
                    
                    # Filtrar para os últimos X anos
                    min_date = pd.Timestamp.now() - pd.DateOffset(years=years)
                    df = df[df['date'] >= min_date]
                    
                    # Manter apenas as colunas necessárias
                    df = df[['date', 'close']]
                    df.columns = ['date', ticker]
                    
                    if all_data.empty:
                        all_data = df
                    else:
                        all_data = pd.merge(all_data, df, on='date', how='outer')
            except Exception as e:
                st.warning(f"Erro ao obter dados para {ticker}: {str(e)}")
    
    # Ordenar por data e definir data como índice
    if not all_data.empty:
        all_data = all_data.sort_values('date')
        all_data.set_index('date', inplace=True)
        
        # Preencher valores faltantes
        all_data = all_data.ffill().bfill()
        
        # Armazenar no cache
        st.session_state[cache_key] = all_data
    
    return all_data

# Função para estimar retornos e covariância
def estimar_retornos_cov(prices_df):
    # Calcular retornos diários
    returns = prices_df.pct_change().dropna()
    
    # Calcular média anualizada (252 dias úteis)
    mu = returns.mean() * 252
    
    # Calcular matriz de covariância anualizada
    S = returns.cov() * 252
    
    return mu, S

# Função para otimizar a carteira
def otimizar(tickers, perfil, prices_df=None):
    if prices_df is None or prices_df.empty:
        prices_df = get_prices(tickers)
    
    if prices_df.empty:
        return {}, (0, 0, 0)
    
    # Remover colunas com valores ausentes
    prices_df = prices_df.dropna(axis=1)
    
    # Verificar se temos dados suficientes
    if prices_df.shape[1] < 10:
        st.warning(f"Dados insuficientes. Apenas {prices_df.shape[1]} ETFs com dados completos.")
        return {}, (0, 0, 0)
    
    try:
        # Calcular retornos e matriz de covariância
        returns = prices_df.pct_change().dropna()
        
        # Aplicar winsorization para remover outliers extremos
        # Isso ajuda a melhorar a estabilidade do modelo
        winsor_returns = returns.copy()
        for col in winsor_returns.columns:
            winsor_returns[col] = winsor_returns[col].clip(
                lower=winsor_returns[col].quantile(0.05),
                upper=winsor_returns[col].quantile(0.95)
            )
            
        # Usar retornos mais estáveis para cálculo de médias
        mu = winsor_returns.mean() * 252
        
        # Garantir que todos os retornos médios são positivos para ETFs
        # com um ajuste mínimo para evitar retornos negativos
        min_expected_return = 0.02  # 2% como retorno mínimo esperado
        mu = np.maximum(mu, min_expected_return)
        
        # Usar matriz de covariância original para preservar relações de risco
        cov_matrix = returns.cov() * 252
        
        # Garantir que a matriz de covariância é positiva definida
        min_eigenval = np.min(np.linalg.eigvals(cov_matrix))
        if min_eigenval < 0:
            # Adicionar um pequeno valor à diagonal para garantir positividade
            cov_matrix = cov_matrix + np.eye(cov_matrix.shape[0]) * (abs(min_eigenval) + 1e-6)
        
        n_assets = len(mu)
        
        # Definir variáveis e parâmetros para CVXPY
        weights = cp.Variable(n_assets)
        returns_vector = np.array(mu)
        
        # Definir risco alvo baseado no perfil
        if perfil == "Conservador":
            risk_target = None  # Vamos minimizar risco
            # Adicionar restrições mais conservadoras
            constraints = [
                cp.sum(weights) == 1,     # Soma dos pesos = 1
                weights >= 0.04,          # Mínimo 4% por ativo
                weights <= 0.20,          # Máximo 20% por ativo
                # Adicionar restrição de diversificação - pelo menos 6 ativos com peso significativo
                cp.sum(cp.minimum(weights, 0.05)) >= 0.30  # Pelo menos 30% do portfólio em 6+ ativos
            ]
        elif perfil == "Moderado":
            risk_target = None  # Vamos maximizar Sharpe
            constraints = [
                cp.sum(weights) == 1,  # Soma dos pesos = 1
                weights >= 0.04,       # Mínimo 4% por ativo
                weights <= 0.20        # Máximo 20% por ativo
            ]
        else:  # Agressivo
            # Definir um risco alvo maior
            risk_target = 0.20
            constraints = [
                cp.sum(weights) == 1,  # Soma dos pesos = 1
                weights >= 0.04,       # Mínimo 4% por ativo
                weights <= 0.25        # Máximo 25% por ativo para perfil agressivo
            ]
        
        # Realizar a otimização com base no perfil
        if perfil == "Conservador":
            # Para conservador, minimizar combinação de risco e drawdown
            # Usar aproximação de drawdown pelo risco
            prob = cp.Problem(
                cp.Minimize(cp.quad_form(weights, cov_matrix)),
                constraints
            )
        elif perfil == "Moderado":
            # Maximizar Sharpe Ratio
            risk_free_rate = 0.03  # Taxa livre de risco (3%)
            risk = cp.quad_form(weights, cov_matrix)**0.5
            excess_return = weights @ returns_vector - risk_free_rate
            
            # Não podemos maximizar excess_return/risk diretamente, então usamos uma aproximação
            # Vamos maximizar excess_return e adicionar uma restrição de risco
            prob = cp.Problem(
                cp.Maximize(excess_return),
                constraints + [risk <= 0.15]  # Limitar volatilidade a 15%
            )
        else:  # Agressivo
            # Maximizar retorno para um nível de risco alvo
            prob = cp.Problem(
                cp.Maximize(weights @ returns_vector),
                constraints + [cp.quad_form(weights, cov_matrix)**0.5 <= risk_target]
            )
        
        # Resolver o problema
        try:
            prob.solve()
        except Exception as e:
            st.warning(f"Erro na otimização principal: {str(e)}. Tentando solução alternativa...")
            # Tentar solução mais simples se falhar, mas ainda mantendo restrições de min e max
            constraints = [
                cp.sum(weights) == 1,  # Soma dos pesos = 1
                weights >= 0.04,       # Mínimo 4% por ativo
                weights <= 0.20        # Máximo 20% por ativo
            ]
            prob = cp.Problem(
                cp.Minimize(cp.quad_form(weights, cov_matrix)),
                constraints
            )
            prob.solve()
        
        # Verificar se obtivemos uma solução
        if weights.value is None:
            raise Exception("Não foi possível encontrar uma solução ótima")
        
        # Obter pesos e normalizar para garantir restrições
        pesos = np.array(weights.value).flatten()
        pesos = np.clip(pesos, 0.04, 0.20)  # Garantir restrições de 4% a 20%
        pesos = pesos / np.sum(pesos)       # Garantir soma = 1
        
        # Criar dicionário de pesos com tickers
        carteira = {ticker: peso for ticker, peso in zip(returns.columns, pesos)}
        
        # Selecionar os top 10
        top10 = dict(sorted(carteira.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # Normalizar os top 10 para soma = 1
        total = sum(top10.values())
        top10 = {k: v/total for k, v in top10.items()}
        
        # Calcular performance esperada
        selected_weights = np.array([top10.get(ticker, 0) for ticker in returns.columns])
        expected_return = np.sum(returns_vector * selected_weights)
        portfolio_risk = np.sqrt(selected_weights.T @ cov_matrix @ selected_weights)
        sharpe_ratio = expected_return / portfolio_risk if portfolio_risk > 0 else 0
        
        performance = (expected_return, portfolio_risk, sharpe_ratio)
        
        return top10, performance
    except Exception as e:
        st.error(f"Erro na otimização: {str(e)}")
        return _otimizar_alternativo(prices_df, perfil)

# Função alternativa para otimização quando CVXPY não está disponível
def _otimizar_alternativo(prices_df, perfil):
    """
    Implementação simplificada de otimização sem usar CVXPY.
    Usa retornos, volatilidade e correlações para criar uma carteira razoável.
    """
    # Calcular retornos históricos
    returns = prices_df.pct_change().dropna()
    
    # Calcular estatísticas básicas
    mean_returns = returns.mean() * 252  # Anualizados
    volatility = returns.std() * np.sqrt(252)  # Anualizada
    sharpe = mean_returns / volatility
    
    # Selecionar ativos com base no perfil
    selected_assets = pd.DataFrame({
        'ticker': returns.columns,
        'return': mean_returns,
        'volatility': volatility,
        'sharpe': sharpe
    })
    
    if perfil == "Conservador":
        # Ordenar por volatilidade (menor primeiro)
        top10 = selected_assets.sort_values('volatility').head(10)
    elif perfil == "Moderado":
        # Ordenar por Sharpe ratio (maior primeiro)
        top10 = selected_assets.sort_values('sharpe', ascending=False).head(10)
    else:  # Agressivo
        # Ordenar por retorno esperado (maior primeiro)
        top10 = selected_assets.sort_values('return', ascending=False).head(10)
    
    # Atribuir pesos diferenciados com base no perfil
    if perfil == "Conservador":
        # Mais peso para ativos menos voláteis (decrescente, de 20% a 4%)
        weights = np.linspace(0.20, 0.04, 10)
    elif perfil == "Moderado":
        # Peso mais equilibrado mas ainda diferente para cada ativo
        weights = np.linspace(0.16, 0.06, 10)
    else:  # Agressivo
        # Mais peso para ativos com maior retorno (decrescente, de 20% a 4%)
        weights = np.linspace(0.20, 0.04, 10)
    
    # Garantir que a soma seja exatamente 1
    weights = weights / weights.sum()
    
    # Garantir limites mínimo e máximo
    weights = np.clip(weights, 0.04, 0.20)
    
    # Normalizar novamente para garantir soma = 1
    weights = weights / weights.sum()
    
    # Criar dicionário de pesos
    portfolio = {ticker: weight for ticker, weight in zip(top10['ticker'], weights)}
    
    # Calcular performance esperada
    expected_return = (top10['return'] * weights).sum()
    port_volatility = np.sqrt(np.dot(weights, np.dot(returns[top10['ticker']].cov() * 252, weights)))
    sharpe_ratio = expected_return / port_volatility
    
    return portfolio, (expected_return, port_volatility, sharpe_ratio)

# Função para gerar análise de texto com OpenAI
def gerar_texto(carteira, perfil):
    prompt = f"""
    Você é analista. Explique esta carteira de 10 ETFs {carteira}
    para um investidor {perfil}, em até 400 palavras, em português simples.
    """
    
    try:
        # Verificar se a API key está disponível
        if not OPENAI_API_KEY:
            return "Não foi possível gerar a análise: Chave de API do OpenAI não configurada."
        
        # Atualizar para usar a nova API da OpenAI
        try:
            from openai import OpenAI
            
            # Remover possíveis prefixos que causam erros
            clean_api_key = OPENAI_API_KEY
            if clean_api_key.startswith("sk-proj-"):
                # Remover prefixo que pode causar o erro "Incorrect API key provided"
                clean_api_key = clean_api_key.replace("sk-proj-", "sk-")
            
            # Criar cliente com a API key limpa
            client = OpenAI(api_key=clean_api_key)
            
            # Fazer a chamada para a API com o novo formato
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Você é um analista financeiro especializado em ETFs e investimentos."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=800
            )
            
            # Extrair e retornar o texto da resposta
            return response.choices[0].message.content.strip()
        except Exception as api_error:
            # Fallback para a API antiga
            try:
                resp = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                return f"Não foi possível gerar a análise: {str(e)}. Você pode encontrar sua API key em https://platform.openai.com/account/api-keys."
    except Exception as e:
        return f"Não foi possível gerar a análise: {str(e)}"

# Função para gerar relatório PDF
def gerar_pdf(df_aloc, analise_texto, perfil, universo, performance):
    # Importações necessárias para o ReportLab
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm, cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    import io
    from datetime import datetime
    
    # Criar um buffer para o PDF
    buffer = io.BytesIO()
    
    # Configurações de página
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=A4,
        leftMargin=1.5*cm,
        rightMargin=1.5*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )
    
    # Lista para armazenar os elementos do documento
    story = []
    
    # Estilos de texto
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Title'],
        fontSize=16,
        alignment=TA_CENTER,
        spaceAfter=6*mm
    )
    
    subtitle_style = ParagraphStyle(
        'SubtitleStyle',
        parent=styles['Heading2'],
        fontSize=14,
        alignment=TA_CENTER,
        spaceAfter=6*mm
    )
    
    header_style = ParagraphStyle(
        'HeaderStyle',
        parent=styles['Heading3'],
        fontSize=12,
        alignment=TA_LEFT,
        spaceAfter=3*mm
    )
    
    normal_style = ParagraphStyle(
        'NormalStyle',
        parent=styles['Normal'],
        fontSize=10,
        spaceBefore=1*mm,
        spaceAfter=3*mm
    )
    
    # Cabeçalho do documento
    story.append(Paragraph("ETF Blueprint - Relatório de Carteira", title_style))
    story.append(Paragraph(f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}", normal_style))
    story.append(Spacer(1, 5*mm))
    
    # Perfil do investidor
    story.append(Paragraph("Perfil do Investidor", header_style))
    story.append(Paragraph(f"Perfil: <b>{perfil}</b>", normal_style))
    story.append(Paragraph(f"Universo de ETFs: <b>{universo}</b>", normal_style))
    story.append(Spacer(1, 5*mm))
    
    # Carteira otimizada
    story.append(Paragraph("Carteira Otimizada", header_style))
    
    # Dados para a tabela
    table_data = [["ETF", "Alocação (%)"]]
    for index, row in df_aloc.iterrows():
        table_data.append([row['ETF'], f"{row['Alocação (%)']:.2f}%"])
    
    # Estilo da tabela
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
    ])
    
    # Criar tabela
    etf_table = Table(table_data, colWidths=[doc.width*0.6, doc.width*0.3])
    etf_table.setStyle(table_style)
    story.append(etf_table)
    story.append(Spacer(1, 5*mm))
    
    # Performance esperada
    expected_return, volatility, sharpe = performance
    story.append(Paragraph("Performance Esperada", header_style))
    
    # Dados para a tabela de performance
    perf_data = [
        ["Retorno Anual Esperado:", f"{expected_return*100:.2f}%"],
        ["Volatilidade Anual:", f"{volatility*100:.2f}%"],
        ["Índice Sharpe:", f"{sharpe:.2f}"]
    ]
    
    # Criar tabela de performance
    perf_table = Table(perf_data, colWidths=[doc.width*0.6, doc.width*0.3])
    perf_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
        ('BOX', (0, 0), (-1, -1), 1, colors.lightgrey),
    ]))
    story.append(perf_table)
    story.append(Spacer(1, 10*mm))
    
    # Análise da carteira
    story.append(Paragraph("Análise da Carteira", header_style))
    story.append(Paragraph(analise_texto, normal_style))
    
    # Rodapé
    story.append(Spacer(1, 10*mm))
    story.append(Paragraph(f"© {datetime.now().year} ETF Blueprint - Todos os direitos reservados", 
                          ParagraphStyle('Footer', parent=styles['Normal'], alignment=TA_CENTER, fontSize=8)))
    
    # Gerar PDF
    doc.build(story)
    
    # Retornar o conteúdo do buffer
    buffer.seek(0)
    return buffer.getvalue()

# Inicializar session_state
if 'stage' not in st.session_state:
    st.session_state['stage'] = "input"

# Título principal
st.title("ETF Blueprint 📈")
st.markdown("Construção de carteiras otimizadas de ETFs")

# Lógica principal do aplicativo
if st.session_state['stage'] == "input":
    # Formulário para perfil de investidor
    with st.form("perfil_form"):
        st.subheader("Perfil do Investidor")
        
        col1, col2 = st.columns(2)
        
        with col1:
            perfil_risco = st.selectbox(
                "Perfil de Risco",
                ["Conservador", "Moderado", "Agressivo"]
            )
            
            horizonte = st.number_input(
                "Horizonte de Investimento (anos)",
                min_value=1,
                max_value=30,
                value=5,
                step=1
            )
        
        with col2:
            max_drawdown = st.slider(
                "Drawdown Máximo Aceitável (%)",
                min_value=5,
                max_value=40,
                value=15,
                step=5
            )
            
            universo = st.radio(
                "Universo de ETFs",
                ["BR", "EUA", "Ambos"]
            )
        
        submit_button = st.form_submit_button("Gerar Carteira")
        
        if submit_button:
            # Salvar perfil no session_state
            st.session_state['perfil'] = {
                'risco': perfil_risco,
                'horizonte': horizonte,
                'max_drawdown': max_drawdown,
                'universo': universo
            }
            
            # Obter lista de ETFs conforme o universo selecionado
            tickers = filtrar_universo(universo)
            
            # Obter preços históricos
            with st.spinner("Obtendo dados históricos e otimizando carteira..."):
                prices_df = get_prices(tickers, horizonte)
                
                if not prices_df.empty:
                    # Otimizar carteira
                    carteira, performance = otimizar(tickers, perfil_risco, prices_df)
                    
                    if carteira:
                        # Criar DataFrame com a alocação
                        df_aloc = pd.DataFrame({
                            'ETF': list(carteira.keys()),
                            'Alocação (%)': [v * 100 for v in carteira.values()]
                        })
                        
                        # Gerar análise de texto
                        analise = gerar_texto(carteira, perfil_risco)
                        
                        # Salvar resultados
                        st.session_state['portfolio_result'] = {
                            'alocacao': df_aloc,
                            'performance': performance,
                            'analise': analise,
                            'carteira': carteira
                        }
                        
                        # Mudar para a tela de resultados
                        st.session_state['stage'] = "result"
                        st.experimental_rerun()
                    else:
                        st.error("Não foi possível otimizar a carteira. Tente outro universo de ETFs.")
                else:
                    st.error("Não foi possível obter dados históricos. Verifique sua conexão e a chave API.")

elif st.session_state['stage'] == "result":
    # Exibir resultados da carteira otimizada
    
    # Recuperar dados
    df_aloc = st.session_state['portfolio_result']['alocacao']
    performance = st.session_state['portfolio_result']['performance']
    analise = st.session_state['portfolio_result']['analise']
    carteira = st.session_state['portfolio_result']['carteira']
    perfil = st.session_state['perfil']
    
    # Mostrar mensagem de sucesso
    st.success("Carteira otimizada com sucesso!")
    
    # Layout em colunas
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Tabela de alocação
        st.subheader("Alocação Recomendada")
        st.dataframe(df_aloc, height=300)
        
        # Métricas de performance
        expected_return, volatility, sharpe = performance
        
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            st.metric("Retorno Esperado", f"{expected_return*100:.2f}%")
        
        with metrics_col2:
            st.metric("Volatilidade", f"{volatility*100:.2f}%")
        
        with metrics_col3:
            st.metric("Índice Sharpe", f"{sharpe:.2f}")
    
    with col2:
        # Gráfico de pizza
        st.subheader("Distribuição da Carteira")
        
        # Garantir que estamos usando o nome da coluna correto
        fig = px.pie(
            df_aloc,
            values='Alocação (%)',  # Confirmar que este é o nome exato da coluna
            names='ETF',
            title='Composição da Carteira'
        )
        
        # Melhorar a formatação dos textos e labels
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            texttemplate='%{label}<br>%{percent:.1%}',
            hovertemplate='<b>%{label}</b><br>Alocação: %{percent:.2%}'
        )
        
        # Atualizar layout para melhor visualização
        fig.update_layout(
            uniformtext_minsize=12,
            uniformtext_mode='hide',
            legend=dict(font=dict(size=10))
        )
        
        st.plotly_chart(fig)
    
    # Análise textual
    st.subheader("Análise da Carteira")
    st.markdown(f"**Perfil: {perfil['risco']} | Universo: {perfil['universo']} | Horizonte: {perfil['horizonte']} anos**")
    st.markdown(analise)
    
    # Botões para download
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        # Download CSV
        csv = df_aloc.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Baixar CSV",
            csv,
            "carteira_etf.csv",
            "text/csv"
        )
    
    with col2:
        # Download PDF
        pdf_bytes = gerar_pdf(
            df_aloc,
            analise,
            perfil['risco'],
            perfil['universo'],
            performance
        )
        
        st.download_button(
            "Baixar PDF",
            pdf_bytes,
            "relatorio_etf_blueprint.pdf",
            "application/pdf"
        )
    
    with col3:
        # Botão para voltar
        if st.button("← Voltar para Formulário"):
            st.session_state['stage'] = "input"
            st.experimental_rerun() 