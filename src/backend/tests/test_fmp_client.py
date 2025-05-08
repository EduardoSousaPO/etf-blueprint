import os
import pytest
import json
import asyncio
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime

from src.backend.services.fmp_client import FMPClient, ETFInfo, HistoricalPriceResponse

# Dados de simulação para testes
MOCK_ETF_LIST = [
    {
        "symbol": "VTI",
        "name": "Vanguard Total Stock Market ETF",
        "price": 250.42,
        "changesPercentage": 0.75,
        "change": 1.86,
        "dayLow": 248.20,
        "dayHigh": 251.30,
        "yearHigh": 255.05,
        "yearLow": 210.10,
        "marketCap": 85000000000,
        "priceAvg50": 245.30,
        "priceAvg200": 235.80,
        "volume": 1200000,
        "avgVolume": 1500000,
        "exchange": "NYSE",
        "open": 249.50,
        "previousClose": 248.56,
        "eps": None,
        "pe": None,
        "earningsAnnouncement": None,
        "sharesOutstanding": None,
        "timestamp": int(datetime.now().timestamp())
    },
    {
        "symbol": "BND",
        "name": "Vanguard Total Bond Market ETF",
        "price": 72.35,
        "changesPercentage": -0.15,
        "change": -0.11,
        "dayLow": 72.20,
        "dayHigh": 72.50,
        "yearHigh": 75.10,
        "yearLow": 70.05,
        "marketCap": 25000000000,
        "priceAvg50": 72.10,
        "priceAvg200": 72.50,
        "volume": 800000,
        "avgVolume": 950000,
        "exchange": "NASDAQ",
        "open": 72.40,
        "previousClose": 72.46,
        "eps": None,
        "pe": None,
        "earningsAnnouncement": None,
        "sharesOutstanding": None,
        "timestamp": int(datetime.now().timestamp())
    }
]

MOCK_HISTORICAL_PRICES = {
    "symbol": "VTI",
    "historical": [
        {
            "date": "2023-01-01",
            "open": 230.10,
            "high": 232.50,
            "low": 229.80,
            "close": 231.20,
            "adjClose": 231.20,
            "volume": 1500000,
            "unadjustedVolume": 1500000,
            "change": 1.10,
            "changePercent": 0.48,
            "vwap": 231.17,
            "label": "January 01, 23",
            "changeOverTime": 0.0048
        },
        {
            "date": "2023-01-02",
            "open": 231.50,
            "high": 233.80,
            "low": 230.20,
            "close": 233.60,
            "adjClose": 233.60,
            "volume": 1600000,
            "unadjustedVolume": 1600000,
            "change": 2.40,
            "changePercent": 1.04,
            "vwap": 232.53,
            "label": "January 02, 23",
            "changeOverTime": 0.0104
        },
        # Adicione dados históricos suficientes para o cálculo de métricas
        *[{
            "date": f"2023-01-{i+3:02d}",
            "open": 233.0 + i*0.5,
            "high": 235.0 + i*0.5,
            "low": 232.0 + i*0.5,
            "close": 234.0 + i*0.5 * (1 + 0.01 * np.sin(i)),
            "adjClose": 234.0 + i*0.5 * (1 + 0.01 * np.sin(i)),
            "volume": 1500000 + i*10000,
            "unadjustedVolume": 1500000 + i*10000,
            "change": 0.5 * (1 + 0.01 * np.sin(i)),
            "changePercent": 0.2 * (1 + 0.01 * np.sin(i)),
            "vwap": 233.5 + i*0.5,
            "label": f"January {i+3:02d}, 23",
            "changeOverTime": 0.01 * (i+1)
        } for i in range(50)]  # Gerar dados para 50 dias adicionais
    ]
}

@pytest.fixture
def mock_env_fmp_key(monkeypatch):
    """Configurar variável de ambiente para teste"""
    monkeypatch.setenv("FMP_API_KEY", "test_api_key")

@pytest.fixture
def fmp_client(mock_env_fmp_key):
    """Criar cliente FMP para testes"""
    return FMPClient()

@pytest.mark.asyncio
async def test_get_etfs_list(fmp_client):
    """Testa a obtenção da lista de ETFs"""
    
    # Mock da resposta HTTP
    with patch('aiohttp.ClientSession.get') as mock_get:
        # Configurar o mock
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__aenter__.return_value = mock_response
        mock_response.json.return_value = MOCK_ETF_LIST
        mock_get.return_value = mock_response
        
        # Chamar o método
        result = await fmp_client.get_etfs_list()
        
        # Verificar se o método fez a chamada correta
        mock_get.assert_called_once()
        assert "etf/list" in mock_get.call_args[0][0]
        
        # Verificar o resultado
        assert len(result) == 2
        assert isinstance(result[0], ETFInfo)
        assert result[0].symbol == "VTI"
        assert result[1].symbol == "BND"

@pytest.mark.asyncio
async def test_get_historical_price(fmp_client):
    """Testa a obtenção de preços históricos"""
    
    # Mock da resposta HTTP
    with patch('aiohttp.ClientSession.get') as mock_get:
        # Configurar o mock
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__aenter__.return_value = mock_response
        mock_response.json.return_value = MOCK_HISTORICAL_PRICES
        mock_get.return_value = mock_response
        
        # Chamar o método
        result = await fmp_client.get_historical_price("VTI", period="1month")
        
        # Verificar se o método fez a chamada correta
        mock_get.assert_called_once()
        assert "historical-price-full/VTI" in mock_get.call_args[0][0]
        
        # Verificar o resultado
        assert isinstance(result, HistoricalPriceResponse)
        assert result.symbol == "VTI"
        assert len(result.historical) > 0
        assert result.historical[0].date == "2023-01-01"
        assert result.historical[1].date == "2023-01-02"

@pytest.mark.asyncio
async def test_calc_etf_metrics(fmp_client):
    """Testa o cálculo de métricas de ETF"""
    
    # Mock do método get_historical_price
    with patch.object(fmp_client, 'get_historical_price') as mock_get_historical:
        # Configurar o mock
        mock_get_historical.return_value = HistoricalPriceResponse.parse_obj(MOCK_HISTORICAL_PRICES)
        
        # Chamar o método
        result = await fmp_client.calc_etf_metrics("VTI")
        
        # Verificar se o método fez a chamada correta
        mock_get_historical.assert_called_once_with("VTI", period="5years")
        
        # Verificar o resultado
        assert isinstance(result, dict)
        assert 'retorno_anualizado' in result
        assert 'volatilidade' in result
        assert 'max_drawdown' in result
        assert 'sharpe_ratio' in result
        
        # Valores devem ser coerentes
        assert isinstance(result['retorno_anualizado'], float)
        assert isinstance(result['volatilidade'], float)
        assert result['max_drawdown'] <= 0  # Drawdown é sempre negativo ou zero

@pytest.mark.asyncio
async def test_calc_correlation_matrix(fmp_client):
    """Testa o cálculo da matriz de correlação"""
    
    # Mock do método get_historical_price
    with patch.object(fmp_client, 'get_historical_price') as mock_get_historical:
        # Configurar o mock para retornar os mesmos dados para ambos ETFs
        mock_get_historical.side_effect = [
            HistoricalPriceResponse.parse_obj(MOCK_HISTORICAL_PRICES),
            HistoricalPriceResponse.parse_obj({
                "symbol": "BND",
                "historical": MOCK_HISTORICAL_PRICES["historical"]
            })
        ]
        
        # Chamar o método
        result = await fmp_client.calc_correlation_matrix(["VTI", "BND"])
        
        # Verificar se o método fez as chamadas corretas
        assert mock_get_historical.call_count == 2
        
        # Verificar o resultado
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 2)  # Matriz 2x2
        assert "VTI" in result.columns
        assert "BND" in result.columns
        assert result.loc["VTI", "VTI"] == 1.0  # Autocorrelação é sempre 1
        assert result.loc["BND", "BND"] == 1.0  # Autocorrelação é sempre 1
        
        # Neste caso, como usamos os mesmos dados, a correlação deve ser 1
        assert result.loc["VTI", "BND"] == 1.0
        assert result.loc["BND", "VTI"] == 1.0

@pytest.mark.asyncio
async def test_error_handling(fmp_client):
    """Testa o tratamento de erros da API"""
    
    # Mock da resposta HTTP com erro
    with patch('aiohttp.ClientSession.get') as mock_get:
        # Configurar o mock
        mock_response = MagicMock()
        mock_response.status = 401
        mock_response.__aenter__.return_value = mock_response
        mock_response.text.return_value = "API Key incorreta"
        mock_get.return_value = mock_response
        
        # Chamar o método e verificar se a exceção é lançada
        with pytest.raises(Exception) as excinfo:
            await fmp_client.get_etfs_list()
        
        # Verificar a mensagem de erro
        assert "Erro na API FMP: 401" in str(excinfo.value) 