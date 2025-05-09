import pandas as pd
import numpy as np
import streamlit as st
from pypfopt import EfficientFrontier, expected_returns, risk_models
from pypfopt.discrete_allocation import DiscreteAllocation
import plotly.express as px
import plotly.graph_objects as go

def estimar_retornos_cov(prices_df):
    """
    Estima retornos esperados e matriz de covariância a partir de preços históricos.
    
    Args:
        prices_df (pd.DataFrame): DataFrame com preços históricos, indexado por data.
        
    Returns:
        tuple: (mu, S) onde mu é o vetor de retornos esperados e S é a matriz de covariância.
    """
    # Calcular retornos diários
    returns = prices_df.pct_change().dropna()
    
    # Calcular média anualizada (252 dias úteis)
    mu = returns.mean() * 252
    
    # Calcular matriz de covariância anualizada
    S = returns.cov() * 252
    
    return mu, S

def otimizar(tickers, perfil, prices_df=None):
    """
    Otimiza uma carteira de ETFs com base no perfil do investidor.
    
    Args:
        tickers (list): Lista de tickers para otimizar.
        perfil (str): Perfil do investidor ('Conservador', 'Moderado', 'Agressivo').
        prices_df (pd.DataFrame, optional): DataFrame com preços históricos. Se None, será obtido.
        
    Returns:
        tuple: (pesos, performance) onde pesos é um dicionário com as alocações e 
               performance é uma tupla (retorno, volatilidade, sharpe).
    """
    # Se não recebeu preços, retorna vazio
    if prices_df is None or prices_df.empty:
        return {}, (0, 0, 0)
    
    # Remover colunas com valores ausentes
    prices_df = prices_df.dropna(axis=1)
    
    # Verificar se ainda temos ativos suficientes
    if prices_df.shape[1] < 10:
        st.warning(f"Dados insuficientes! Apenas {prices_df.shape[1]} ativos têm dados completos.")
        return {}, (0, 0, 0)
    
    # Calcular retornos esperados e covariância
    mu = expected_returns.mean_historical_return(prices_df)
    S = risk_models.sample_cov(prices_df)
    
    # Criar o objeto EfficientFrontier
    ef = EfficientFrontier(mu, S)
    
    # Adicionar restrições
    ef.add_constraint(lambda w: w >= 0.04)  # Mínimo 4% por ativo
    ef.add_constraint(lambda w: w <= 0.20)  # Máximo 20% por ativo
    
    # Otimizar de acordo com o perfil
    if perfil == "Conservador":
        ef.min_volatility()
    elif perfil == "Moderado":
        ef.max_sharpe()
    else:  # Agressivo
        ef.efficient_risk(target_risk=0.20)
    
    # Limpar os pesos
    pesos = ef.clean_weights()
    
    # Selecionar os top 10 ETFs com maiores pesos
    top10 = dict(sorted(pesos.items(), key=lambda x: x[1], reverse=True)[:10])
    
    # Normalizar para soma = 1
    total = sum(top10.values())
    top10 = {k: v/total for k, v in top10.items()}
    
    # Calcular performance
    performance = ef.portfolio_performance(verbose=False)
    
    return top10, performance

def generate_efficient_frontier_plot(prices_df, optimal_weights, risk_free_rate=0.01):
    """
    Gera um gráfico da fronteira eficiente.
    
    Args:
        prices_df (pd.DataFrame): DataFrame com preços históricos.
        optimal_weights (dict): Dicionário com pesos ótimos da carteira.
        risk_free_rate (float): Taxa livre de risco.
        
    Returns:
        plotly.graph_objects.Figure: Figura do Plotly com o gráfico da fronteira eficiente.
    """
    # Calcular retornos e covariância
    mu = expected_returns.mean_historical_return(prices_df)
    S = risk_models.sample_cov(prices_df)
    
    # Gerar pontos da fronteira eficiente
    ef = EfficientFrontier(mu, S)
    
    # Gerar carteiras aleatórias para visualização
    n_samples = 1000
    weights = np.random.dirichlet(np.ones(len(mu)), n_samples)
    
    # Calcular retorno e volatilidade para cada carteira
    ret = []
    vol = []
    sharpe = []
    
    for i in range(n_samples):
        w = weights[i]
        r = (w * mu).sum()
        v = np.sqrt(np.dot(w, np.dot(S, w)))
        s = (r - risk_free_rate) / v
        ret.append(r)
        vol.append(v)
        sharpe.append(s)
    
    # Criar DataFrame para visualização
    df = pd.DataFrame({
        'Retorno': ret,
        'Volatilidade': vol,
        'Sharpe': sharpe
    })
    
    # Calcular retorno e volatilidade da carteira otimizada
    optimal_return = sum(w * mu[ticker] for ticker, w in optimal_weights.items())
    optimal_volatility = np.sqrt(
        sum(w1 * w2 * S.loc[ticker1, ticker2]
            for ticker1, w1 in optimal_weights.items()
            for ticker2, w2 in optimal_weights.items())
    )
    
    # Criar figura com Plotly
    fig = px.scatter(
        df, x='Volatilidade', y='Retorno', 
        color='Sharpe', color_continuous_scale='Viridis',
        title='Fronteira Eficiente de Markowitz'
    )
    
    # Adicionar carteira otimizada
    fig.add_trace(
        go.Scatter(
            x=[optimal_volatility],
            y=[optimal_return],
            mode='markers',
            marker=dict(size=15, color='red', symbol='star'),
            name='Carteira Otimizada'
        )
    )
    
    # Ajustar layout
    fig.update_layout(
        xaxis_title='Volatilidade Anual',
        yaxis_title='Retorno Anual Esperado',
        coloraxis_colorbar=dict(title='Índice Sharpe'),
        showlegend=True,
        height=500
    )
    
    # Formatar eixos como percentual
    fig.update_xaxes(tickformat='.0%')
    fig.update_yaxes(tickformat='.0%')
    
    return fig 