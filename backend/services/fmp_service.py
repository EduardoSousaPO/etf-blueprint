import requests
import pandas as pd
import streamlit as st
import os
import time
from datetime import datetime, timedelta

# Obter chave API da FMP
FMP_API_KEY = os.getenv("FMP_API_KEY", st.secrets.get("FMP_API_KEY", ""))

def get_prices(tickers, years=5):
    """
    Obtém preços históricos para uma lista de tickers usando a API FMP.
    
    Args:
        tickers (list): Lista de símbolos/tickers.
        years (int): Número de anos de histórico para obter.
        
    Returns:
        pd.DataFrame: DataFrame com preços históricos, indexado por data.
    """
    # Verificar se já temos esses dados em cache
    cache_key = f"prices_cache_{','.join(tickers)}_{years}"
    
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    
    # DataFrame para armazenar todos os dados
    all_data = pd.DataFrame()
    
    with st.spinner(f"Obtendo dados históricos para {len(tickers)} ETFs..."):
        for ticker in tickers:
            try:
                # Construir URL para a API FMP
                url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={FMP_API_KEY}"
                
                # Realizar a requisição
                response = requests.get(url)
                data = response.json()
                
                # Processar se tiver dados históricos
                if "historical" in data:
                    # Converter para DataFrame
                    df = pd.DataFrame(data["historical"])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date')
                    
                    # Filtrar para os últimos X anos
                    min_date = pd.Timestamp.now() - pd.DateOffset(years=years)
                    df = df[df['date'] >= min_date]
                    
                    # Manter apenas as colunas necessárias (data e preço de fechamento)
                    df = df[['date', 'close']]
                    df.columns = ['date', ticker]
                    
                    # Adicionar ao DataFrame principal
                    if all_data.empty:
                        all_data = df
                    else:
                        all_data = pd.merge(all_data, df, on='date', how='outer')
                else:
                    st.warning(f"Sem dados históricos para {ticker}")
                
                # Pequena pausa para evitar atingir limites de rate da API
                time.sleep(0.1)
                
            except Exception as e:
                st.warning(f"Erro ao obter dados para {ticker}: {str(e)}")
    
    # Processar o DataFrame resultante
    if not all_data.empty:
        # Ordenar por data e definir data como índice
        all_data = all_data.sort_values('date')
        all_data.set_index('date', inplace=True)
        
        # Preencher valores faltantes
        all_data = all_data.ffill().bfill()
        
        # Armazenar no cache do session_state
        st.session_state[cache_key] = all_data
    
    return all_data

def get_etf_info(ticker):
    """
    Obtém informações detalhadas sobre um ETF específico.
    
    Args:
        ticker (str): Símbolo/ticker do ETF.
        
    Returns:
        dict: Informações sobre o ETF.
    """
    try:
        # Cache key para evitar chamadas repetidas
        cache_key = f"etf_info_{ticker}"
        
        if cache_key in st.session_state:
            return st.session_state[cache_key]
        
        # Construir URL para a API FMP
        url = f"https://financialmodelingprep.com/api/v3/etf/profile/{ticker}?apikey={FMP_API_KEY}"
        
        # Realizar a requisição
        response = requests.get(url)
        data = response.json()
        
        if data and isinstance(data, list) and len(data) > 0:
            etf_info = data[0]
            st.session_state[cache_key] = etf_info
            return etf_info
        else:
            return None
            
    except Exception as e:
        st.warning(f"Erro ao obter informações para o ETF {ticker}: {str(e)}")
        return None 