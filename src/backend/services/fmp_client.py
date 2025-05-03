import os
import json
import aiohttp
import asyncio
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import numpy as np
from pydantic import BaseModel, Field, validator
import statistics
import math

# Modelos Pydantic para validação
class ETFInfo(BaseModel):
    symbol: str
    name: Optional[str] = None
    price: Optional[float] = None
    changesPercentage: Optional[float] = None
    change: Optional[float] = None
    dayLow: Optional[float] = None
    dayHigh: Optional[float] = None
    yearHigh: Optional[float] = None
    yearLow: Optional[float] = None
    marketCap: Optional[float] = None
    priceAvg50: Optional[float] = None
    priceAvg200: Optional[float] = None
    volume: Optional[int] = None
    avgVolume: Optional[int] = None
    exchange: Optional[str] = None
    open: Optional[float] = None
    previousClose: Optional[float] = None
    eps: Optional[float] = None
    pe: Optional[float] = None
    earningsAnnouncement: Optional[str] = None
    sharesOutstanding: Optional[int] = None
    timestamp: Optional[int] = None

class HistoricalPrice(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    adjClose: float
    volume: int
    unadjustedVolume: Optional[int] = None
    change: Optional[float] = None
    changePercent: Optional[float] = None
    vwap: Optional[float] = None
    label: str
    changeOverTime: Optional[float] = None

class HistoricalPriceResponse(BaseModel):
    symbol: str
    historical: List[HistoricalPrice]

class FMPClient:
    """
    Cliente para a API Financial Modeling Prep (FMP)
    """
    
    BASE_URL = "https://financialmodelingprep.com/api/v3"
    
    def __init__(self, api_key: str = None):
        """
        Inicializa o cliente FMP
        
        Args:
            api_key: Chave de API FMP. Se não for fornecida, tenta ler de FMP_API_KEY
        """
        self.api_key = api_key or os.getenv("FMP_API_KEY")
        if not self.api_key:
            raise ValueError("FMP API Key não encontrada. Defina FMP_API_KEY ou passe como parâmetro.")
        
        # Cache para minimizar chamadas repetidas
        self._cache = {}
        # TTL padrão de 24 horas para cache
        self._cache_ttl = 24 * 60 * 60
    
    async def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict:
        """
        Realiza uma requisição à API FMP
        
        Args:
            endpoint: Endpoint da API, sem barra inicial
            params: Parâmetros adicionais para a requisição
            
        Returns:
            Resposta da API como dict
        """
        # Preparar parâmetros
        params = params or {}
        params["apikey"] = self.api_key
        
        # Verificar cache
        cache_key = f"{endpoint}:{json.dumps(params, sort_keys=True)}"
        if cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if datetime.now().timestamp() - cached_time < self._cache_ttl:
                return cached_data
        
        # Construir URL
        url = f"{self.BASE_URL}/{endpoint}"
        
        print(f"Fazendo requisição para: {url}")
        print(f"Parâmetros: {params}")
        
        # Fazer requisição
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        error_msg = f"Erro na API FMP: {response.status} - {error_text}"
                        print(error_msg)
                        raise Exception(error_msg)
                    
                    # Obter resposta como texto para análise
                    response_text = await response.text()
                    
                    # Tentar converter para JSON
                    try:
                        data = json.loads(response_text)
                    except json.JSONDecodeError as e:
                        print(f"Resposta não é um JSON válido: {response_text[:500]}...")
                        raise
                    
                    # Verificar se a resposta é vazia ou um array vazio
                    if data is None or (isinstance(data, list) and len(data) == 0):
                        error_msg = f"API FMP retornou dados vazios para {endpoint}"
                        print(error_msg)
                        raise Exception(error_msg)
                    
                    # Verificar se a resposta contém uma mensagem de erro
                    if isinstance(data, dict) and "error" in data:
                        error_msg = f"Erro na API FMP: {data['error']}"
                        print(error_msg)
                        raise Exception(error_msg)
                    
                    # Salvar no cache
                    self._cache[cache_key] = (datetime.now().timestamp(), data)
                    
                    print(f"Resposta recebida de {url} com sucesso")
                    if isinstance(data, list):
                        print(f"Número de itens: {len(data)}")
                        if len(data) > 0:
                            print(f"Primeiro item (amostra): {json.dumps(data[0], indent=2)[:500]}...")
                    
                    return data
        except aiohttp.ClientError as e:
            error_msg = f"Erro de conexão com a API FMP: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)
        except json.JSONDecodeError as e:
            error_msg = f"Erro ao decodificar resposta da API FMP: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Erro inesperado na API FMP: {str(e)}"
            print(error_msg)
            raise
    
    async def get_etfs_list(self) -> List[ETFInfo]:
        """
        Obtém lista de todos os ETFs disponíveis
        
        Returns:
            Lista de informações de ETFs
        """
        response = await self._make_request("etf/list")
        if not response or len(response) == 0:
            raise Exception("API FMP não retornou dados para a lista de ETFs")
        
        # Adicionar logs para diagnóstico
        print(f"Total de ETFs recebidos: {len(response)}")
        print(f"Exemplo do primeiro ETF recebido: {response[0]}")
        
        # Tentar fazer o parsing de cada ETF, pulando os que não têm a estrutura correta
        etfs_info = []
        for etf in response:
            try:
                etfs_info.append(ETFInfo.parse_obj(etf))
            except Exception as e:
                print(f"Erro ao processar ETF {etf.get('symbol', 'desconhecido')}: {str(e)}")
                continue
        
        print(f"Total de ETFs processados com sucesso: {len(etfs_info)}")
        
        if len(etfs_info) == 0:
            raise Exception("Nenhum ETF pôde ser processado corretamente da resposta da API")
            
        return etfs_info
    
    async def get_historical_price(
        self, 
        symbol: str, 
        from_date: Optional[str] = None, 
        to_date: Optional[str] = None,
        period: str = "5years"
    ) -> HistoricalPriceResponse:
        """
        Obtém preços históricos para um ETF
        
        Args:
            symbol: Símbolo do ETF
            from_date: Data inicial no formato 'YYYY-MM-DD' (opcional)
            to_date: Data final no formato 'YYYY-MM-DD' (opcional)
            period: Período ('1month', '3month', '6month', '1year', '5years')
        
        Returns:
            Dados históricos de preços
        """
        endpoint = f"historical-price-full/{symbol}"
        params = {}
        
        # Calcular datas com base no período se não forem fornecidas
        if not from_date or not to_date:
            today = datetime.now()
            
            # Definir a data final como hoje se não fornecida
            if not to_date:
                to_date = today.strftime('%Y-%m-%d')
            
            # Calcular from_date com base no período se não fornecida
            if not from_date:
                if period == "5years":
                    from_date = (today - timedelta(days=5*365)).strftime('%Y-%m-%d')
                elif period == "1year":
                    from_date = (today - timedelta(days=365)).strftime('%Y-%m-%d')
                elif period == "6month":
                    from_date = (today - timedelta(days=182)).strftime('%Y-%m-%d')
                elif period == "3month":
                    from_date = (today - timedelta(days=91)).strftime('%Y-%m-%d')
                elif period == "1month":
                    from_date = (today - timedelta(days=30)).strftime('%Y-%m-%d')
                else:
                    # Período padrão: 5 anos
                    from_date = (today - timedelta(days=5*365)).strftime('%Y-%m-%d')
        
        # Usar datas explícitas para a consulta
        params["from"] = from_date
        params["to"] = to_date
        
        print(f"Buscando dados históricos para {symbol} de {from_date} até {to_date}")
        
        try:
            # Fazer a requisição com datas explícitas
            response = await self._make_request(endpoint, params)
            
            # Verificar se o response tem a estrutura esperada
            if not isinstance(response, dict) or "symbol" not in response or "historical" not in response:
                raise ValueError(f"Formato de resposta inesperado para {symbol}")
                
            # Verificar se temos dados históricos suficientes
            if len(response.get("historical", [])) < 100:
                print(f"AVISO: Poucos dados históricos ({len(response.get('historical', []))}) para {symbol}.")
                
                # Vamos tentar usar o parâmetro timeseries como fallback se temos poucos dados
                if len(response.get("historical", [])) < 10:
                    print(f"Tentando abordagem alternativa com timeseries para {symbol}...")
                    alt_params = {"timeseries": period}
                    try:
                        alt_response = await self._make_request(endpoint, alt_params)
                        if isinstance(alt_response, dict) and "historical" in alt_response:
                            if len(alt_response["historical"]) > len(response.get("historical", [])):
                                print(f"Abordagem alternativa retornou mais dados: {len(alt_response['historical'])} pontos")
                                response = alt_response
                    except Exception as e:
                        print(f"Tentativa alternativa com timeseries falhou: {str(e)}")
            
            return HistoricalPriceResponse.parse_obj(response)
        
        except Exception as e:
            print(f"Erro ao obter dados históricos para {symbol}: {str(e)}")
            # Recriar o objeto com dados mínimos para evitar falha completa
            return HistoricalPriceResponse(
                symbol=symbol,
                historical=[]
            )
    
    async def calc_etf_metrics(self, symbol: str, period: str = "5years") -> Dict[str, float]:
        """
        Calcula métricas de um ETF com base nos dados históricos
        
        Args:
            symbol: Símbolo do ETF
            period: Período para análise
            
        Returns:
            Dict com métricas: retorno_anualizado, volatilidade, max_drawdown
        """
        # Definir retornos históricos e volatilidades por classe de ativos
        # Valores baseados em médias históricas de mercado de longo prazo (20+ anos)
        benchmark_returns = {
            'us_large_cap': 9.8,       # Ações americanas large cap (S&P 500)
            'us_mid_cap': 10.4,        # Ações americanas mid cap
            'us_small_cap': 11.3,      # Ações americanas small cap
            'international_developed': 8.5,  # Mercados desenvolvidos ex-US
            'international_emerging': 10.2,  # Mercados emergentes
            'us_treasury': 4.2,        # Títulos do tesouro americano
            'us_corporate_bonds': 5.8, # Títulos corporativos
            'us_high_yield': 7.2,      # High yield bonds
            'us_tips': 4.0,            # TIPS (Treasury Inflation Protected)
            'real_estate': 9.5,        # REITs
            'commodities': 6.0,        # Commodities diversificadas
            'gold': 5.5,               # Ouro
            'tech': 12.5,              # Tecnologia
            'healthcare': 11.0,        # Saúde
            'financial': 9.8,          # Financeiro
            'consumer': 9.5,           # Consumo
            'energy': 9.0,             # Energia
            'utilities': 8.5,          # Utilities
            'dividend': 9.2,           # Ações de dividendos
            'value': 9.6,              # Ações de valor
            'growth': 10.2,            # Ações de crescimento
            'momentum': 10.5,          # Ações de momentum
            'low_volatility': 9.0,     # Baixa volatilidade
            'crypto': 15.0,            # Criptomoedas (altamente especulativo)
            'global_balanced': 8.5,    # Portfolio global balanceado
        }
        
        # Volatilidades típicas por classe de ativos (desvio padrão anual, %)
        benchmark_volatilities = {
            'us_large_cap': 15.0,
            'us_mid_cap': 17.0,
            'us_small_cap': 19.5,
            'international_developed': 16.5,
            'international_emerging': 22.0,
            'us_treasury': 5.0,
            'us_corporate_bonds': 7.5,
            'us_high_yield': 10.0,
            'us_tips': 5.5,
            'real_estate': 18.0,
            'commodities': 17.5,
            'gold': 15.5,
            'tech': 22.5,
            'healthcare': 17.0,
            'financial': 19.0,
            'consumer': 15.0,
            'energy': 20.5,
            'utilities': 14.0,
            'dividend': 14.0,
            'value': 15.5,
            'growth': 17.0,
            'momentum': 18.0,
            'low_volatility': 12.0,
            'crypto': 65.0,
            'global_balanced': 10.5,
        }
        
        # Drawdowns históricos máximos por classe de ativos (%)
        benchmark_drawdowns = {
            'us_large_cap': -35.0,
            'us_mid_cap': -40.0,
            'us_small_cap': -45.0,
            'international_developed': -40.0,
            'international_emerging': -50.0,
            'us_treasury': -10.0,
            'us_corporate_bonds': -15.0,
            'us_high_yield': -25.0,
            'us_tips': -12.0,
            'real_estate': -45.0,
            'commodities': -35.0,
            'gold': -30.0,
            'tech': -42.0,
            'healthcare': -30.0,
            'financial': -48.0,
            'consumer': -32.0,
            'energy': -40.0,
            'utilities': -25.0,
            'dividend': -30.0,
            'value': -35.0,
            'growth': -38.0,
            'momentum': -35.0,
            'low_volatility': -25.0,
            'crypto': -75.0,
            'global_balanced': -28.0,
        }
        
        # Mapeamento de ETFs para suas classes de ativos
        etf_asset_class = {
            # ETFs de índices amplos dos EUA
            'SPY': 'us_large_cap',   # SPDR S&P 500 ETF Trust
            'IVV': 'us_large_cap',   # iShares Core S&P 500 ETF
            'VOO': 'us_large_cap',   # Vanguard S&P 500 ETF
            'VTI': 'us_large_cap',   # Vanguard Total Stock Market ETF
            'QQQ': 'tech',           # Invesco QQQ Trust (Nasdaq-100)
            'IWM': 'us_small_cap',   # iShares Russell 2000 ETF
            'MDY': 'us_mid_cap',     # SPDR S&P MidCap 400 ETF
            'IJH': 'us_mid_cap',     # iShares Core S&P Mid-Cap ETF
            
            # ETFs internacionais
            'EFA': 'international_developed',  # iShares MSCI EAFE ETF
            'VEA': 'international_developed',  # Vanguard FTSE Developed Markets ETF
            'EEM': 'international_emerging',   # iShares MSCI Emerging Markets ETF
            'VWO': 'international_emerging',   # Vanguard FTSE Emerging Markets ETF
            'IEMG': 'international_emerging',  # iShares Core MSCI Emerging Markets ETF
            
            # ETFs de títulos (bonds)
            'AGG': 'us_corporate_bonds',  # iShares Core U.S. Aggregate Bond ETF
            'BND': 'us_corporate_bonds',  # Vanguard Total Bond Market ETF
            'LQD': 'us_corporate_bonds',  # iShares iBoxx $ Investment Grade Corporate Bond ETF
            'HYG': 'us_high_yield',       # iShares iBoxx $ High Yield Corporate Bond ETF
            'JNK': 'us_high_yield',       # SPDR Bloomberg High Yield Bond ETF
            'TLT': 'us_treasury',         # iShares 20+ Year Treasury Bond ETF
            'IEF': 'us_treasury',         # iShares 7-10 Year Treasury Bond ETF
            'SHY': 'us_treasury',         # iShares 1-3 Year Treasury Bond ETF
            'TIP': 'us_tips',             # iShares TIPS Bond ETF
            
            # ETFs de setores
            'XLK': 'tech',        # Technology Select Sector SPDR Fund
            'XLF': 'financial',   # Financial Select Sector SPDR Fund
            'XLV': 'healthcare',  # Health Care Select Sector SPDR Fund
            'XLE': 'energy',      # Energy Select Sector SPDR Fund
            'XLU': 'utilities',   # Utilities Select Sector SPDR Fund
            'XLP': 'consumer',    # Consumer Staples Select Sector SPDR Fund
            'XLY': 'consumer',    # Consumer Discretionary Select Sector SPDR Fund
            
            # ETFs de estilo de investimento
            'VTV': 'value',           # Vanguard Value ETF
            'IVE': 'value',           # iShares S&P 500 Value ETF
            'VUG': 'growth',          # Vanguard Growth ETF
            'IVW': 'growth',          # iShares S&P 500 Growth ETF
            'MTUM': 'momentum',       # iShares MSCI USA Momentum Factor ETF
            'USMV': 'low_volatility', # iShares MSCI USA Min Vol Factor ETF
            'VIG': 'dividend',        # Vanguard Dividend Appreciation ETF
            'HDV': 'dividend',        # iShares Core High Dividend ETF
            
            # ETFs de real estate
            'VNQ': 'real_estate',  # Vanguard Real Estate ETF
            'IYR': 'real_estate',  # iShares U.S. Real Estate ETF
            
            # ETFs de commodities
            'GLD': 'gold',         # SPDR Gold Shares
            'IAU': 'gold',         # iShares Gold Trust
            'DBC': 'commodities',  # Invesco DB Commodity Index Tracking Fund
            'GSG': 'commodities',  # iShares S&P GSCI Commodity-Indexed Trust
            
            # ETFs de alocação (multi-ativos)
            'AOR': 'global_balanced',  # iShares Core Growth Allocation ETF
            'AOA': 'global_balanced',  # iShares Core Aggressive Allocation ETF
            'AOK': 'global_balanced',  # iShares Core Conservative Allocation ETF
            'AOM': 'global_balanced',  # iShares Core Moderate Allocation ETF
            
            # ETFs relacionados a criptomoedas
            'BITO': 'crypto',  # ProShares Bitcoin Strategy ETF
            'GBTC': 'crypto',  # Grayscale Bitcoin Trust
        }
        
        # Obter a classe de ativo para o símbolo, ou usar "us_large_cap" como padrão
        asset_class = etf_asset_class.get(symbol, 'us_large_cap')
        
        try:
            # Buscar dados históricos
            hist_data = await self.get_historical_price(symbol, period=period)
            historical = hist_data.historical
            
            # Contar quantos pontos de dados temos
            num_data_points = len(historical)
            print(f"Obtidos {num_data_points} pontos de dados históricos para {symbol}")
            
            # Verificar se temos dados suficientes para cálculos confiáveis
            # Precisamos de pelo menos 100 pontos de dados para cálculos estatísticos confiáveis
            if num_data_points < 100:
                print(f"AVISO: Dados históricos insuficientes para {symbol} ({num_data_points} pontos). Usando benchmarks.")
                # Usar valores de benchmark se não temos dados suficientes
                return {
                    "retorno_anualizado": benchmark_returns.get(asset_class, 9.0),
                    "volatilidade": benchmark_volatilities.get(asset_class, 15.0),
                    "max_drawdown": benchmark_drawdowns.get(asset_class, -30.0)
                }
            
            # Se temos dados, calcular métricas
            # Converter dados para um DataFrame para análise mais fácil
            dates = [pd.to_datetime(item.date) for item in historical]
            closes = [float(item.close) for item in historical]
            
            # Ordenar por data (mais antiga primeiro)
            date_close = sorted(zip(dates, closes), key=lambda x: x[0])
            dates = [d[0] for d in date_close]
            closes = [d[1] for d in date_close]
            
            # Calcular o período total em anos
            if len(dates) >= 2:
                dias_no_periodo = (dates[-1] - dates[0]).days
                anos_no_periodo = dias_no_periodo / 365.25
                print(f"Período analisado para {symbol}: {anos_no_periodo:.2f} anos")
                
                # Para evitar problemas com períodos muito curtos
                if anos_no_periodo < 0.5:  # Menos de 6 meses
                    print(f"AVISO: Período muito curto para {symbol} ({anos_no_periodo:.2f} anos). Usando valores mistos.")
                    return {
                        "retorno_anualizado": benchmark_returns.get(asset_class, 9.0) * 0.7 + (closes[-1]/closes[0] - 1) * 100 * 0.3,
                        "volatilidade": benchmark_volatilities.get(asset_class, 15.0),
                        "max_drawdown": benchmark_drawdowns.get(asset_class, -30.0)
                    }
            else:
                # Não temos dados suficientes para calcular o período
                print(f"AVISO: Dados insuficientes para {symbol}. Usando benchmarks.")
                return {
                    "retorno_anualizado": benchmark_returns.get(asset_class, 9.0),
                    "volatilidade": benchmark_volatilities.get(asset_class, 15.0),
                    "max_drawdown": benchmark_drawdowns.get(asset_class, -30.0)
                }
            
            # Calcular retorno total no período e anualizar
            retorno_total = (closes[-1] / closes[0] - 1) * 100  # Em porcentagem
            retorno_anualizado = retorno_total / anos_no_periodo
            print(f"Retorno total de {symbol}: {retorno_total:.2f}% em {anos_no_periodo:.2f} anos")
            
            # Calcular retornos diários
            retornos_diarios = []
            for i in range(1, len(closes)):
                daily_return = (closes[i] / closes[i-1] - 1) * 100  # Em porcentagem
                retornos_diarios.append(daily_return)
            
            # Calcular volatilidade (desvio padrão dos retornos)
            volatilidade_diaria = statistics.stdev(retornos_diarios) if len(retornos_diarios) > 1 else 0
            # Anualizar a volatilidade (multiplicar pela raiz quadrada do número de dias de trading em um ano)
            volatilidade_anualizada = volatilidade_diaria * math.sqrt(252)
            
            # Calcular drawdown máximo
            max_drawdown = 0
            peak = closes[0]
            for close in closes:
                if close > peak:
                    peak = close
                drawdown = (close / peak - 1) * 100  # Em porcentagem
                max_drawdown = min(max_drawdown, drawdown)
            
            # Balancear com os valores de benchmark para evitar outliers extremos
            # (especialmente para períodos de dados curtos)
            peso_dados_reais = min(1.0, anos_no_periodo / 5.0)  # Peso total para dados de 5+ anos
            peso_benchmark = 1.0 - peso_dados_reais
            
            # Aplicar pesos para misturar dados reais com benchmarks
            retorno_anualizado_final = (retorno_anualizado * peso_dados_reais + 
                                      benchmark_returns.get(asset_class, 9.0) * peso_benchmark)
            
            volatilidade_final = (volatilidade_anualizada * peso_dados_reais + 
                                benchmark_volatilities.get(asset_class, 15.0) * peso_benchmark)
            
            max_drawdown_final = (max_drawdown * peso_dados_reais + 
                                benchmark_drawdowns.get(asset_class, -30.0) * peso_benchmark)
            
            # Aplicar limites para evitar valores irrealistas
            retorno_anualizado_final = max(-5, min(25, retorno_anualizado_final))
            volatilidade_final = max(3, min(35, volatilidade_final))
            max_drawdown_final = max(-75, min(-5, max_drawdown_final))
            
            print(f"Métricas finais para {symbol}: Retorno={retorno_anualizado_final:.2f}%, Vol={volatilidade_final:.2f}%, MaxDD={max_drawdown_final:.2f}%")
            
            return {
                "retorno_anualizado": retorno_anualizado_final,
                "volatilidade": volatilidade_final,
                "max_drawdown": max_drawdown_final
            }
            
        except Exception as e:
            print(f"Erro ao calcular métricas para {symbol}: {str(e)}")
            # Em caso de erro, usar valores de benchmark
            return {
                "retorno_anualizado": benchmark_returns.get(asset_class, 9.0),
                "volatilidade": benchmark_volatilities.get(asset_class, 15.0),
                "max_drawdown": benchmark_drawdowns.get(asset_class, -30.0)
            }
    
    async def calc_correlation_matrix(self, symbols: List[str], period: str = "5years") -> pd.DataFrame:
        """
        Calcula matriz de correlação entre ETFs
        
        Args:
            symbols: Lista de símbolos de ETFs
            period: Período para análise
            
        Returns:
            DataFrame com matriz de correlação
        """
        # Obter dados de todos os ETFs
        tasks = [self.get_historical_price(symbol, period=period) for symbol in symbols]
        historical_data = await asyncio.gather(*tasks)
        
        # Criar DataFrame consolidado
        all_prices = {}
        
        for i, data in enumerate(historical_data):
            symbol = symbols[i]
            prices = pd.DataFrame([{
                'date': h.date,
                'close': h.adjClose
            } for h in data.historical])
            
            prices['date'] = pd.to_datetime(prices['date'])
            all_prices[symbol] = prices.set_index('date')['close']
        
        # Juntar todos os preços em um DataFrame
        price_df = pd.DataFrame(all_prices)
        
        # Calcular retornos diários
        returns_df = price_df.pct_change().dropna()
        
        # Calcular matriz de correlação
        correlation_matrix = returns_df.corr()
        
        return correlation_matrix 