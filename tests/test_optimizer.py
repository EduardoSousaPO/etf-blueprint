import sys
import os
import pandas as pd
import numpy as np
import pytest

# Adicionar diretório pai ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar função de otimização
from backend.services.optimizer import otimizar

def test_otimizacao_formato():
    """
    Testa se a função de otimização retorna um formato correto.
    """
    # Criar dados de teste
    np.random.seed(42)
    
    # Criar DataFrame de preços simulados
    dates = pd.date_range(start='2020-01-01', periods=252*3, freq='B')
    tickers = [f'STOCK_{i}' for i in range(20)]
    
    # Gerar preços aleatórios
    data = {}
    for ticker in tickers:
        # Gerar série de preços com tendência leve
        returns = np.random.normal(0.0005, 0.01, len(dates))
        prices = 100 * (1 + returns).cumprod()
        data[ticker] = prices
    
    # Criar DataFrame
    df = pd.DataFrame(data, index=dates)
    
    # Testar otimização para diferentes perfis
    for perfil in ["Conservador", "Moderado", "Agressivo"]:
        carteira, performance = otimizar(tickers, perfil, df)
        
        # Verificar se a carteira tem 10 ativos
        assert len(carteira) == 10, f"A carteira deve ter exatamente 10 ativos para o perfil {perfil}"
        
        # Verificar se a soma dos pesos é aproximadamente 1
        assert 0.999 <= sum(carteira.values()) <= 1.001, f"A soma dos pesos deve ser 1 para o perfil {perfil}"
        
        # Verificar se os pesos estão entre 4% e 20%
        for ticker, peso in carteira.items():
            assert 0.04 <= peso <= 0.20, f"O peso de {ticker} deve estar entre 4% e 20% para o perfil {perfil}"
        
        # Verificar o formato da performance
        assert len(performance) == 3, "Performance deve retornar (retorno, volatilidade, sharpe)"
        expected_return, volatility, sharpe = performance
        
        # Valores de retorno e volatilidade devem ser positivos
        assert expected_return > 0, f"Retorno esperado deve ser positivo para o perfil {perfil}"
        assert volatility > 0, f"Volatilidade deve ser positiva para o perfil {perfil}"

def test_otimizacao_diferenciacao():
    """
    Testa se a otimização produz resultados diferentes para perfis diferentes.
    """
    # Criar dados de teste
    np.random.seed(42)
    
    # Criar DataFrame de preços simulados
    dates = pd.date_range(start='2020-01-01', periods=252*3, freq='B')
    tickers = [f'STOCK_{i}' for i in range(20)]
    
    # Gerar preços aleatórios com comportamentos diferentes
    data = {}
    for i, ticker in enumerate(tickers):
        # Variar tendência e volatilidade para criar ativos diversos
        mu = 0.0005 + (i % 5) * 0.0002  # Diferentes médias
        sigma = 0.01 + (i % 3) * 0.002  # Diferentes volatilidades
        
        returns = np.random.normal(mu, sigma, len(dates))
        prices = 100 * (1 + returns).cumprod()
        data[ticker] = prices
    
    # Criar DataFrame
    df = pd.DataFrame(data, index=dates)
    
    # Obter carteiras para diferentes perfis
    conservador, _ = otimizar(tickers, "Conservador", df)
    moderado, _ = otimizar(tickers, "Moderado", df)
    agressivo, _ = otimizar(tickers, "Agressivo", df)
    
    # Verificar se as carteiras são diferentes
    assert conservador != moderado, "As carteiras Conservador e Moderado devem ser diferentes"
    assert moderado != agressivo, "As carteiras Moderado e Agressivo devem ser diferentes"
    assert conservador != agressivo, "As carteiras Conservador e Agressivo devem ser diferentes"

if __name__ == "__main__":
    # Executar os testes manualmente
    test_otimizacao_formato()
    test_otimizacao_diferenciacao()
    print("Todos os testes passaram!") 