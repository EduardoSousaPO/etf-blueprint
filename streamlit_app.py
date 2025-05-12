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
import scipy.stats

# Verificar se CVXPY está disponível
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

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

# Função segura para acessar secrets e env vars
def get_api_key(key_name, default=""):
    """Busca chave de API com fallback seguro."""
    try:
        # Tentar pegar do secrets primeiro
        from_secrets = st.secrets.get("api_keys", {}).get(key_name, "")
        if from_secrets:
            return from_secrets
    except Exception:
        pass  # Ignora erros e continua
    
    # Fallback para variáveis de ambiente
    return os.getenv(key_name, default)

# Chaves API com acesso seguro
FMP_API_KEY = get_api_key("FMP_API_KEY")
OPENAI_API_KEY = get_api_key("OPENAI_API_KEY")

# Configurar OpenAI apenas se a chave estiver disponível
if OPENAI_API_KEY:
    try:
        openai.api_key = OPENAI_API_KEY
    except Exception:
        pass  # Ignora erros na configuração da API

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

# Função para obter preços históricos modificada com fallback para dados simulados
def get_prices(tickers, start_date=None, end_date=None):
    """
    Obtém preços históricos para os tickers fornecidos.
    Retorna um DataFrame com os preços.
    """
    # Verificar se a API KEY da FMP está disponível
    fmp_api_key = get_api_key("FMP_API_KEY")
    
    # Se não tiver API KEY ou estiver vazia, usar dados simulados
    if not fmp_api_key:
        return generate_simulated_prices(tickers)
    
    # Tentar obter dados da API
    try:
        all_data = pd.DataFrame()
        for ticker in tickers:
            try:
                url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={fmp_api_key}"
                response = requests.get(url, timeout=10)  # Adicionar timeout
                data = response.json()
                
                if 'historical' not in data:
                    continue
                
                df = pd.DataFrame(data['historical'])
                df['ticker'] = ticker
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                
                all_data = pd.concat([all_data, df])
            except Exception as e:
                st.warning(f"Erro ao obter dados para {ticker}: {e}")
                continue
        
        if all_data.empty:
            return generate_simulated_prices(tickers)
        
        all_data = all_data.pivot(columns='ticker', values='adjClose')
        return all_data
    except Exception as e:
        st.warning(f"Erro ao obter dados históricos: {e}")
        return generate_simulated_prices(tickers)

# Função para gerar dados simulados quando a API não estiver disponível
def generate_simulated_prices(tickers, days=252*2):
    """
    Gera preços simulados para os tickers fornecidos.
    Retorna um DataFrame com os preços simulados.
    """
    np.random.seed(42)  # Para reprodutibilidade
    
    # Criar datas para os últimos 2 anos (aproximadamente 252 dias úteis por ano)
    end_date = datetime.now()  # Corrigido: usar datetime.now() ao invés de datetime.datetime.now()
    dates = [end_date - datetime.timedelta(days=i) for i in range(days)]
    dates.reverse()
    
    simulated_data = pd.DataFrame(index=dates)
    
    for ticker in tickers:
        # Definir preço inicial entre 50 e 200
        initial_price = np.random.uniform(50, 200)
        
        # Simular uma tendência com volatilidade realista
        returns = np.random.normal(0.0003, 0.015, days)  # média positiva pequena, desvio 1.5%
        prices = [initial_price]
        
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        simulated_data[ticker] = prices[1:]  # Remover o preço inicial extra
    
    simulated_data.index = pd.DatetimeIndex(dates)
    return simulated_data

# Função para estimar retornos e covariância
def estimar_retornos_cov(prices_df):
    # Calcular retornos diários
    returns = prices_df.pct_change().dropna()
    
    # Calcular média anualizada (252 dias úteis)
    mu = returns.mean() * 252
    
    # Calcular matriz de covariância anualizada
    S = returns.cov() * 252
    
    return mu, S

# Função para otimizar a carteira com melhor tratamento de erros
def otimizar(tickers, perfil, prices_df=None):
    """Otimiza uma carteira de ETFs conforme o perfil de risco."""
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
    
    # Se não temos CVXPY disponível, usar implementação alternativa
    if not CVXPY_AVAILABLE:
        return _otimizar_alternativo(prices_df, perfil)
    
    try:
        # Calcular retornos e matriz de covariância
        returns = prices_df.pct_change().dropna()
        
        # Aplicar winsorization para remover outliers extremos
        winsor_returns = returns.copy()
        for col in winsor_returns.columns:
            try:
                winsor_returns[col] = scipy.stats.mstats.winsorize(winsor_returns[col], limits=[0.01, 0.01])
            except Exception:
                # Se falhar, manter os valores originais
                pass
        
        # Calcular retornos esperados e matriz de covariância
        expected_returns = winsor_returns.mean() * 252  # Anualizado
        cov_matrix = winsor_returns.cov() * 252  # Anualizada

        # Garantir que os retornos esperados não sejam todos negativos
        if (expected_returns <= 0).all():
            # Adicionar um pequeno valor positivo para garantir que existam retornos positivos
            expected_returns = expected_returns + abs(expected_returns.min()) + 0.02

        # Número de ativos
        n_assets = len(expected_returns)
        
        # Definir variáveis de peso
        weights = cp.Variable(n_assets)
        
        # Restrições comuns para todos os perfis
        constraints = [
            cp.sum(weights) == 1,  # Soma dos pesos = 1
            weights >= 0.04,        # Mínimo de 4% por ativo
            weights <= 0.20         # Máximo de 20% por ativo
        ]

        # Escolher objetivo e restrições adicionais com base no perfil
        if perfil == "Conservador":
            # Minimizar risco (variância do portfólio)
            risk = cp.quad_form(weights, cov_matrix)
            objective = cp.Minimize(risk)
            
            # Adicionar restrição de retorno mínimo para evitar portfólio muito defensivo
            constraints.append(expected_returns @ weights >= 0.05)  # Retorno mínimo de 5%
            
        elif perfil == "Moderado":
            # Maximizar Sharpe Ratio (retorno / risco)
            risk = cp.quad_form(weights, cov_matrix)
            ret = expected_returns @ weights
            # Como não podemos maximizar ret/risk diretamente, fixamos o denominador
            objective = cp.Maximize(ret)
            constraints.append(risk <= 0.1)  # Risco moderado (10% volatilidade anual)
            
        else:  # "Agressivo"
            # Maximizar retorno com restrição de risco
            ret = expected_returns @ weights
            risk = cp.quad_form(weights, cov_matrix)
            objective = cp.Maximize(ret)
            constraints.append(risk <= 0.2)  # Permitir risco maior (20% volatilidade anual)
        
        # Resolver o problema de otimização
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve()
            
            if prob.status not in ["infeasible", "unbounded"]:
                # Obter pesos ótimos
                optimal_weights = weights.value
                
                # Criar dicionário de alocação
                allocation = {}
                for i, ticker in enumerate(expected_returns.index):
                    if optimal_weights[i] > 0.001:  # Ignorar pesos muito pequenos
                        allocation[ticker] = round(optimal_weights[i] * 100, 2)
                
                # Calcular métricas do portfólio otimizado
                portfolio_return = float(expected_returns @ optimal_weights)
                portfolio_risk = float(np.sqrt(optimal_weights @ cov_matrix @ optimal_weights))
                portfolio_sharpe = float(portfolio_return / portfolio_risk if portfolio_risk > 0 else 0)
                
                return allocation, (portfolio_return, portfolio_risk, portfolio_sharpe)
            else:
                # Se o problema for inviável ou ilimitado, usar método alternativo
                return _otimizar_alternativo(prices_df, perfil)
        except Exception as e:
            st.warning(f"Erro na otimização: {e}")
            return _otimizar_alternativo(prices_df, perfil)
            
    except Exception as e:
        st.warning(f"Erro na preparação dos dados: {e}")
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

# Adicionar função para calcular fronteira eficiente (após a função _otimizar_alternativo)
def calcular_fronteira_eficiente(prices_df, n_portfolios=50):
    """Calcula a fronteira eficiente com tratamento de erros e tipos melhorado."""
    # Verificar se temos dados suficientes
    if prices_df.shape[1] < 2:
        raise ValueError("São necessários pelo menos 2 ativos para calcular a fronteira eficiente")
    
    # Calcular retornos diários
    returns = prices_df.pct_change().dropna()
    
    # Verificar se temos dias suficientes
    if returns.shape[0] < 20:
        raise ValueError("São necessários pelo menos 20 dias de dados para calcular a fronteira eficiente")
    
    # Número de ativos
    n_assets = returns.shape[1]
    
    # Arrays para armazenar dados dos portfólios
    all_weights = np.zeros((n_portfolios, n_assets))
    ret_arr = np.zeros(n_portfolios)
    vol_arr = np.zeros(n_portfolios)
    sharpe_arr = np.zeros(n_portfolios)
    
    # Calcular médias de retornos anualizados
    mean_returns = returns.mean() * 252
    
    # Garantir que haja retornos positivos
    if (mean_returns <= 0).all():
        mean_returns = mean_returns + abs(mean_returns.min()) + 0.02
    
    # Calcular matriz de covariância anualizada
    cov_matrix = returns.cov() * 252
    
    # Taxa livre de risco
    risk_free = 0.03  # 3% ao ano
    
    # Gerar portfólios aleatórios
    for i in range(n_portfolios):
        # Gerar pesos aleatórios
        weights = np.random.random(n_assets)
        weights = weights / np.sum(weights)
        
        # Restringir aos limites (4%-20%)
        weights = np.clip(weights, 0.04, 0.20)
        
        # Garantir que a soma seja 1
        weights = weights / np.sum(weights)
        
        # Garantir que o número de ativos com peso >= 4% não exceda o número total de ativos
        max_assets = min(n_assets, 25)  # Limitar a 25 ativos no máximo
        nonzero_weights = weights[weights >= 0.04]
        
        if len(nonzero_weights) > max_assets:
            # Manter apenas os top pesos
            cutoff = np.sort(weights)[-max_assets]
            weights[weights < cutoff] = 0
            weights = weights / np.sum(weights)
        
        # Salvar pesos
        all_weights[i, :] = weights
        
        # Calcular retorno esperado
        ret_arr[i] = np.sum(mean_returns * weights)
        
        # Calcular volatilidade esperada
        vol_arr[i] = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Calcular Sharpe ratio
        sharpe_arr[i] = (ret_arr[i] - risk_free) / vol_arr[i] if vol_arr[i] > 0 else 0
    
    # Combinar resultados
    frontier_data = {
        'Retorno': ret_arr,
        'Volatilidade': vol_arr,
        'Sharpe': sharpe_arr
    }
    
    return pd.DataFrame(frontier_data)

# Modificar a função gerar_texto para ser mais robusta
def gerar_texto(carteira, perfil):
    """Gera análise textual da carteira usando OpenAI API com melhor tratamento de erros."""
    prompt = f"""
    Você é analista financeiro. Explique esta carteira de ETFs {carteira}
    para um investidor {perfil}, em até 300 palavras, em português simples.
    Foque em explicar a diversificação e estratégia da carteira.
    """
    
    # Se não temos API key, usar o texto demo
    if not OPENAI_API_KEY:
        return gerar_texto_demo(carteira, perfil)
    
    try:
        # Tentar a nova API da OpenAI primeiro
        try:
            from openai import OpenAI
            
            # Criar cliente com a API key
            client = OpenAI(api_key=OPENAI_API_KEY)
            
            # Fazer a chamada para a API com o modelo mais barato
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # Modelo mais barato
                messages=[
                    {"role": "system", "content": "Você é um analista financeiro especializado em ETFs. Seja direto e objetivo."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            # Extrair e retornar o texto da resposta
            return response.choices[0].message.content.strip()
        except Exception as api_error:
            # Tentar alternativa com API antiga
            try:
                resp = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Você é um analista financeiro especializado em ETFs. Seja direto e objetivo."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
                return resp.choices[0].message.content.strip()
            except Exception:
                # Se falhar, usar o modo de demonstração
                return gerar_texto_demo(carteira, perfil)
    except Exception:
        # Se ocorrer qualquer erro, usar o modo de demonstração
        return gerar_texto_demo(carteira, perfil)

# Função para gerar texto de análise com OpenAI (após a função gerar_texto existente)
def gerar_texto_demo(carteira, perfil):
    """
    Gera um texto de análise para o modo de demonstração, sem usar a API OpenAI.
    """
    etfs_list = list(carteira.keys())
    alocacao_list = list(carteira.values())
    
    if perfil == "Conservador":
        return f"""Esta carteira otimizada de {len(etfs_list)} ETFs está alinhada com seu perfil conservador, priorizando a preservação de capital e estabilidade. 
        
A diversificação entre {', '.join(etfs_list[:3])} e outros ETFs ajuda a reduzir a volatilidade e proporcionar um crescimento consistente ao longo do tempo.

A alocação está balanceada para minimizar o risco, com maior peso em {etfs_list[0]} ({alocacao_list[0]}%) e {etfs_list[1]} ({alocacao_list[1]}%), que historicamente apresentam menor volatilidade.

Recomenda-se revisão semestral da carteira para pequenos ajustes conforme as condições de mercado."""
    
    elif perfil == "Moderado":
        return f"""Esta carteira de {len(etfs_list)} ETFs foi otimizada para seu perfil moderado, buscando um equilíbrio entre crescimento e proteção patrimonial.
        
Com maior alocação em {etfs_list[0]} ({alocacao_list[0]}%) e {etfs_list[1]} ({alocacao_list[1]}%), a carteira combina instrumentos de maior potencial de valorização com outros mais estáveis.

A diversificação entre diferentes classes de ativos proporciona uma exposição balanceada ao mercado, adequada para um horizonte de investimento de médio prazo.

Recomenda-se revisão trimestral da carteira para ajustes que mantenham o alinhamento com seus objetivos financeiros."""
    
    else:  # Agressivo
        return f"""Esta carteira de {len(etfs_list)} ETFs está alinhada com seu perfil agressivo, focando em maximizar retornos com maior tolerância à volatilidade.
        
A alocação dá preferência a {etfs_list[0]} ({alocacao_list[0]}%) e {etfs_list[1]} ({alocacao_list[1]}%), ETFs com maior potencial de valorização, complementados por outros instrumentos para diversificação estratégica.

Esta composição busca capturar oportunidades de crescimento significativo no longo prazo, aceitando oscilações de mercado no curto e médio prazo.

Recomenda-se revisão mensal ou bimestral da carteira para potencialmente aumentar a exposição em segmentos com momentum favorável."""

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

# Código principal do aplicativo (substitua a seção existente após as funções)
if __name__ == "__main__":
    # Interface do usuário
    st.set_page_config(page_title="ETF Blueprint 📈", page_icon="📈")
    
    # Título e descrição
    st.title("ETF Blueprint 📈")
    st.write("Construção de carteiras otimizadas de ETFs")
    
    # Verificar estado da sessão
    if 'stage' not in st.session_state:
        st.session_state['stage'] = "input"
    
    # Formulário de entrada
    if st.session_state['stage'] == "input":
        with st.form("portfolio_form"):
            st.subheader("Perfil do Investidor")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Perfil de Risco")
                perfil = st.selectbox(
                    "Escolha seu perfil",
                    options=["Conservador", "Moderado", "Agressivo"],
                    key="perfil"
                )
            
            with col2:
                st.write("Drawdown Máximo Aceitável (%)")
                drawdown = st.slider(
                    "",
                    min_value=5,
                    max_value=40,
                    value=15,
                    key="drawdown"
                )
            
            st.write("Horizonte de Investimento (anos)")
            horizon = st.slider(
                "",
                min_value=1,
                max_value=10,
                value=5,
                key="horizon"
            )
            
            st.write("Universo de ETFs")
            universo = st.radio(
                "",
                options=["BR", "EUA", "Ambos"],
                horizontal=True,
                key="universo"
            )
            
            submitted = st.form_submit_button("Gerar Carteira")
            
            if submitted:
                # Verificar se temos API key da FMP
                fmp_api_key = get_api_key("FMP_API_KEY")
                if not fmp_api_key:
                    # Usar dados simulados (modo demonstração)
                    st.session_state['demo_mode'] = True
                    progress_text = "Gerando carteira otimizada com dados simulados (modo demonstração)..."
                else:
                    st.session_state['demo_mode'] = False
                    progress_text = "Gerando carteira otimizada..."
                
                progress_bar = st.progress(0)
                progress_status = st.empty()
                progress_status.write(progress_text)
                
                # Selecionar ETFs com base no universo escolhido
                if universo == "BR":
                    etfs = ETFS_BR
                elif universo == "EUA":
                    etfs = ETFS_EUA
                else:
                    etfs = ETFS_BR + ETFS_EUA
                
                # Processar dados e otimizar carteira
                try:
                    progress_status.write(f"{progress_text} Obtendo dados de preços...")
                    progress_bar.progress(0.2)
                    
                    # Obter dados de preço (históricos ou simulados)
                    prices_df = get_prices(etfs)
                    # Guardar para uso na fronteira eficiente
                    st.session_state['precos_historicos'] = prices_df
                    
                    progress_status.write(f"{progress_text} Otimizando alocação...")
                    progress_bar.progress(0.6)
                    
                    # Otimizar portfólio
                    allocation, performance = otimizar(etfs, perfil, prices_df)
                    
                    # Criar dataframe de alocação
                    df_aloc = pd.DataFrame(allocation.items(), columns=['ETF', 'Alocação (%)'])
                    df_aloc = df_aloc.sort_values('Alocação (%)', ascending=False)
                    
                    progress_status.write(f"{progress_text} Gerando análise textual...")
                    progress_bar.progress(0.8)
                    
                    # Gerar texto de análise
                    if st.session_state.get('demo_mode', False):
                        analise = gerar_texto_demo(allocation, perfil)
                    else:
                        analise = gerar_texto(allocation, perfil)
                    
                    progress_status.write(f"{progress_text} Concluindo...")
                    progress_bar.progress(1.0)
                    
                    # Armazenar resultados na sessão
                    st.session_state['portfolio_result'] = {
                        'alocacao': df_aloc,
                        'performance': performance,
                        'analise': analise,
                        'carteira': allocation,
                        'perfil': perfil,
                        'universo': universo
                    }
                    
                    # Avançar para a tela de resultados
                    st.session_state['stage'] = "result"
                    
                    # Recarregar a página para exibir resultados
                    st.experimental_rerun()
                    
                except Exception as e:
                    st.error(f"Ocorreu um erro durante a otimização: {str(e)}")

    # Mostrar resultados da carteira otimizada
    elif st.session_state['stage'] == "result":
        # Exibir resultados da carteira otimizada
        
        # Recuperar dados
        df_aloc = st.session_state['portfolio_result']['alocacao']
        performance = st.session_state['portfolio_result']['performance']
        analise = st.session_state['portfolio_result']['analise']
        carteira = st.session_state['portfolio_result']['carteira']
        perfil = st.session_state['portfolio_result']['perfil']
        universo = st.session_state['portfolio_result']['universo']
        
        # Verificar se estamos em modo demonstração
        if st.session_state.get('demo_mode', False):
            st.info("**Modo demonstração:** Esta carteira foi gerada com dados simulados, pois não há uma chave de API configurada.")
        
        # Mostrar mensagem de sucesso
        st.success("Carteira otimizada com sucesso!")
        
        # Layout em colunas
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Tabela de alocação
            st.subheader("Alocação Recomendada")
            st.dataframe(df_aloc, use_container_width=True)
            
            # Métricas de desempenho
            st.subheader("Desempenho Esperado")
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            
            retorno, risco, sharpe = performance
            
            perf_col1.metric(
                "Retorno Anual Esperado",
                f"{retorno:.2%}"
            )
            
            perf_col2.metric(
                "Volatilidade Anual",
                f"{risco:.2%}"
            )
            
            perf_col3.metric(
                "Sharpe Ratio",
                f"{sharpe:.2f}"
            )
            
            # Análise textual
            st.subheader("Análise da Carteira")
            st.markdown(analise)
            
            # Adicionar gráfico de fronteira eficiente
            st.subheader("Fronteira Eficiente")
            
            # Se temos os preços históricos no session_state, usar para calcular fronteira
            if 'precos_historicos' in st.session_state:
                prices_df = st.session_state['precos_historicos']
            else:
                # Selecionar ETFs com base no universo escolhido
                if universo == "BR":
                    etfs = ETFS_BR
                elif universo == "EUA":
                    etfs = ETFS_EUA
                else:
                    etfs = ETFS_BR + ETFS_EUA
                # Obter dados de preço
                prices_df = get_prices(etfs)
                # Guardar para uso futuro
                st.session_state['precos_historicos'] = prices_df
            
            # Calcular fronteira eficiente
            try:
                frontier_df = calcular_fronteira_eficiente(prices_df)
                
                # Criar gráfico de fronteira eficiente
                fig_ef = go.Figure()
                
                # Adicionar pontos da fronteira
                fig_ef.add_trace(
                    go.Scatter(
                        x=frontier_df['Volatilidade'], 
                        y=frontier_df['Retorno'],
                        mode='markers',
                        marker=dict(
                            size=6,
                            color=frontier_df['Sharpe'],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Sharpe Ratio")
                        ),
                        name='Portfólios Possíveis'
                    )
                )
                
                # Adicionar a carteira otimizada
                fig_ef.add_trace(
                    go.Scatter(
                        x=[risco],
                        y=[retorno],
                        mode='markers',
                        marker=dict(
                            size=15,
                            color='red',
                            symbol='star'
                        ),
                        name='Carteira Otimizada'
                    )
                )
                
                # Atualizar layout
                fig_ef.update_layout(
                    title='Fronteira Eficiente',
                    xaxis_title='Volatilidade (Risco)',
                    yaxis_title='Retorno Esperado',
                    xaxis=dict(tickformat=".1%"),
                    yaxis=dict(tickformat=".1%"),
                    height=350,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )
                
                st.plotly_chart(fig_ef, use_container_width=True)
            except Exception as e:
                st.warning(f"Não foi possível gerar o gráfico de fronteira eficiente: {str(e)}")
            
            # Botões de download
            btn_col1, btn_col2 = st.columns(2)
            
            with btn_col1:
                # Preparar CSV para download
                csv = df_aloc.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Baixar CSV",
                    data=csv,
                    file_name=f"etf_portfolio_{perfil.lower()}.csv",
                    mime="text/csv",
                    key="download-csv"
                )
            
            with btn_col2:
                # Preparar PDF para download
                pdf_file = gerar_pdf(df_aloc, analise, perfil, universo, performance)
                st.download_button(
                    label="Baixar PDF",
                    data=pdf_file,
                    file_name=f"etf_portfolio_{perfil.lower()}.pdf",
                    mime="application/pdf",
                    key="download-pdf"
                )
        
        with col2:
            # Gráfico de pizza
            st.subheader("Distribuição da Carteira")
            
            # Clonamos os dados para garantir que não haja problemas de referência
            plot_data = df_aloc.copy()
            
            # Garantir que os dados são do tipo correto
            plot_data['Alocação (%)'] = plot_data['Alocação (%)'].astype(float)
            
            # Criar o gráfico usando uma abordagem mais direta
            try:
                fig = px.pie(
                    plot_data,
                    values="Alocação (%)",  # Nome exato da coluna
                    names="ETF",
                    title='Composição da Carteira'
                )
                
                # Configurações para melhorar a exibição
                fig.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    texttemplate="%{label}<br>%{percent:.1%}",
                    hovertemplate='<b>%{label}</b><br>Alocação: %{value:.2f}%<extra></extra>'
                )
                
                # Ajustar layout
                fig.update_layout(
                    showlegend=False,  # Sem legenda para ficar mais limpo
                    height=400,
                    margin=dict(t=40, b=0, l=0, r=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Erro ao gerar gráfico de pizza: {e}")
        
        # Botão para retornar
        if st.button("← Voltar para Entrada", key="back-btn"):
            st.session_state['stage'] = "input"
            st.experimental_rerun() 