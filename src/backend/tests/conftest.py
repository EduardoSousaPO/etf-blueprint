"""
Configurações compartilhadas para testes da aplicação ETF Blueprint
"""

import os
import pytest
from unittest.mock import patch
import asyncio


# Fixture para definir variáveis de ambiente para testes
@pytest.fixture(scope="session", autouse=True)
def setup_env_vars():
    """Define variáveis de ambiente necessárias para testes"""
    os.environ["FMP_API_KEY"] = "test_api_key"
    os.environ["OPENAI_API_KEY"] = "test_openai_key"
    
    # Garantir que os testes rodam em modo não interativo
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    
    yield
    
    # Limpar variáveis de ambiente após os testes
    del os.environ["FMP_API_KEY"]
    del os.environ["OPENAI_API_KEY"]
    del os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"]
    del os.environ["STREAMLIT_SERVER_HEADLESS"]


# Fixture para loop de eventos asyncio
@pytest.fixture(scope="session")
def event_loop():
    """Cria um loop de eventos para testes assíncronos"""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close() 