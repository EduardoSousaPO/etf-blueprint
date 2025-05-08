import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy.optimize import minimize
import asyncio
from src.backend.services.fmp_client import FMPClient
from dataclasses import dataclass

@dataclass
class ETFData:
    """Dados de um ETF para otimização"""
    symbol: str
    name: str
    categoria: str
    expense_ratio: float
    retorno_anual: float
    volatilidade: float

@dataclass
class CarteiraOtimizada:
    """Resultado da otimização de carteira"""
    etfs: List[Dict[str, Any]]
    pesos: List[float]
    retorno_esperado: float
    volatilidade: float
    sharpe_ratio: float
    max_drawdown: float
    correlacao_matriz: Optional[pd.DataFrame] = None

class PortfolioOptimizer:
    """
    Serviço de otimização de carteira utilizando o algoritmo de Markowitz (Mean-Variance Optimization)
    """
    
    def __init__(self, fmp_client: Optional[FMPClient] = None, risk_free_rate: float = 0.02):
        """
        Inicializa o otimizador de carteira
        
        Args:
            fmp_client: Cliente FMP opcional (se não fornecido, será criado internamente)
            risk_free_rate: Taxa livre de risco anual (default 2%)
        """
        self.fmp_client = fmp_client
        self.risk_free_rate = risk_free_rate
    
    async def _ensure_fmp_client(self):
        """
        Garante que temos um cliente FMP disponível
        """
        if self.fmp_client is None:
            self.fmp_client = FMPClient()
    
    async def get_etf_data(self, etf_symbols: List[str], period: str = "5years") -> List[ETFData]:
        """
        Obtém dados de ETFs necessários para otimização
        
        Args:
            etf_symbols: Lista de símbolos de ETFs
            period: Período para análise de histórico
            
        Returns:
            Lista de objetos ETFData com informações dos ETFs
        """
        await self._ensure_fmp_client()
        
        # Dados estáticos para taxa de despesa e categorias dos ETFs especificados
        expense_ratios = {
            # ETFs solicitados pelo usuário
            'SMH': 0.35, 'SCHG': 0.04, 'MGK': 0.07, 'FTEC': 0.08, 'VGT': 0.10, 
            'IWY': 0.18, 'IGV': 0.42, 'QQQ': 0.20, 'QQQM': 0.15, 'XLK': 0.10, 
            'VONG': 0.08, 'CGGR': 0.45, 'XLG': 0.20, 'VOOG': 0.10, 'SPYG': 0.04, 
            'OEF': 0.20, 'CIBR': 0.60, 'JEPQ': 0.35, 'MGC': 0.07, 'SCHB': 0.03, 
            'IOO': 0.40, 'ESGV': 0.12, 'QUAL': 0.19, 'VOO': 0.03, 'IVV': 0.03, 
            'VONE': 0.10, 'IWB': 0.15, 'ESGU': 0.15, 'ITOT': 0.03, 'IWP': 0.18, 
            'GSLC': 0.09, 'DSI': 0.25, 'SPLG': 0.03, 'SPY': 0.09, 'SPHQ': 0.15, 
            'XMHQ': 0.30, 'PAVE': 0.25, 'MTUM': 0.15, 'CGDV': 0.50, 'AVUS': 0.15, 
            'VFH': 0.10, 'DUHP': 0.29, 'VIS': 0.10, 'VT': 0.07, 'CGGO': 0.50, 
            'SDVY': 0.50, 'FNDX': 0.25, 'RDVY': 0.50, 'BUFR': 0.32
        }
        
        # Mapeamento de categorias
        categorias = {
            # ETFs setoriais
            'SMH': 'Setorial - Tecnologia', 'VGT': 'Setorial - Tecnologia', 'FTEC': 'Setorial - Tecnologia',
            'XLK': 'Setorial - Tecnologia', 'IGV': 'Setorial - Software', 'CIBR': 'Setorial - Cibersegurança',
            'VFH': 'Setorial - Financeiro', 'VIS': 'Setorial - Industrial', 'PAVE': 'Setorial - Infraestrutura',
            
            # ETFs de crescimento
            'SCHG': 'Ações EUA - Crescimento', 'MGK': 'Ações EUA - Crescimento', 
            'IWY': 'Ações EUA - Crescimento', 'VONG': 'Ações EUA - Crescimento', 
            'VOOG': 'Ações EUA - Crescimento', 'SPYG': 'Ações EUA - Crescimento',
            'CGGR': 'Ações EUA - Crescimento',
            
            # ETFs de tecnologia/qualidade
            'QQQ': 'Ações EUA - Nasdaq', 'QQQM': 'Ações EUA - Nasdaq',
            
            # ETFs de mega cap
            'XLG': 'Ações EUA - Mega Cap', 'MGC': 'Ações EUA - Mega Cap',
            
            # ETFs de índice amplo
            'OEF': 'Ações EUA - S&P 100',
            'SCHB': 'Ações EUA - Mercado Total', 'ITOT': 'Ações EUA - Mercado Total',
            'VOO': 'Ações EUA - S&P 500', 'IVV': 'Ações EUA - S&P 500', 
            'SPY': 'Ações EUA - S&P 500', 'SPLG': 'Ações EUA - S&P 500',
            'VONE': 'Ações EUA - Russell 1000', 'IWB': 'Ações EUA - Russell 1000',
            'GSLC': 'Ações EUA - Large Cap',
            'IWP': 'Ações EUA - Mid Cap Crescimento',
            'VT': 'Ações Globais',
            'IOO': 'Ações Globais - Desenvolvidos',
            
            # ETFs ESG
            'ESGV': 'Ações EUA - ESG', 'ESGU': 'Ações EUA - ESG', 'DSI': 'Ações EUA - ESG',
            
            # ETFs de fator
            'QUAL': 'Fator - Qualidade', 'MTUM': 'Fator - Momentum', 
            'SPHQ': 'Fator - Qualidade', 'XMHQ': 'Fator - Qualidade',
            'DUHP': 'Fator - Baixa Volatilidade', 'AVUS': 'Fator - Valor',
            
            # ETFs de renda
            'JEPQ': 'Ações EUA - Income', 
            'CGDV': 'Ações EUA - Dividendos', 'SDVY': 'Ações EUA - Dividendos',
            'RDVY': 'Ações EUA - Dividendos', 'CGGO': 'Ações Globais - Crescimento',
            'FNDX': 'Ações EUA - Fundamentalista',
            
            # ETFs de buffer
            'BUFR': 'Buffer ETF'
        }
        
        # Definir categoria padrão para ETFs não mapeados
        categoria_padrao = "Outro"
        
        print(f"Iniciando obtenção de dados para {len(etf_symbols)} ETFs...")
        
        try:
            # Obter lista completa de ETFs
            print("Buscando lista de ETFs disponíveis na API...")
            etfs_info = await self.fmp_client.get_etfs_list()
            print(f"Lista de ETFs obtida: {len(etfs_info)} ETFs encontrados")
            
            # Criar mapa de símbolo -> info
            etfs_info_map = {etf.symbol: etf for etf in etfs_info}
            print(f"Mapa de ETFs criado com {len(etfs_info_map)} entradas")
            
            # Verificar quais símbolos foram encontrados
            encontrados = [symbol for symbol in etf_symbols if symbol in etfs_info_map]
            nao_encontrados = [symbol for symbol in etf_symbols if symbol not in etfs_info_map]
            
            print(f"ETFs encontrados: {len(encontrados)} de {len(etf_symbols)}")
            if nao_encontrados:
                print(f"ETFs não encontrados: {', '.join(nao_encontrados)}")
            
            # Obter métricas para cada ETF encontrado
            etf_data_list = []
            for symbol in encontrados:
                print(f"Obtendo métricas para {symbol}...")
                
                try:
                    # Obter métricas do ETF (retorno, volatilidade, etc.)
                    metrics = await self.fmp_client.calc_etf_metrics(symbol, period)
                    
                    # Obter nome do ETF do mapa ou usar o símbolo como fallback
                    etf_name = getattr(etfs_info_map[symbol], 'name', f"{symbol} ETF")
                    
                    # Criar objeto ETFData
                    etf_data = ETFData(
                        symbol=symbol,
                        name=etf_name,
                        categoria=categorias.get(symbol, categoria_padrao),
                        expense_ratio=expense_ratios.get(symbol, 0.20),  # Valor padrão se não encontrado
                        retorno_anual=metrics["retorno_anualizado"],
                        volatilidade=metrics["volatilidade"]
                    )
                    
                    etf_data_list.append(etf_data)
                    print(f"ETF {symbol} processado com sucesso: retorno={metrics['retorno_anualizado']:.2f}%, vol={metrics['volatilidade']:.2f}%")
                    
                except Exception as e:
                    print(f"Erro ao processar ETF {symbol}: {str(e)}")
            
            print(f"Total de ETFs processados com sucesso: {len(etf_data_list)}")
            
            if len(etf_data_list) < 4:
                raise ValueError(f"Número insuficiente de ETFs processados ({len(etf_data_list)}). Necessário no mínimo 4 para otimização.")
            
            return etf_data_list
            
        except Exception as e:
            print(f"Erro na obtenção de dados dos ETFs: {str(e)}")
            raise
    
    async def optimize_portfolio(
        self,
        etf_data: List[ETFData],
        target_return: Optional[float] = None,
        risk_profile: str = "Moderado",
        constraints: Optional[Dict[str, Any]] = None,
        max_volatility: Optional[float] = None,
        max_drawdown: Optional[float] = None
    ) -> CarteiraOtimizada:
        """
        Otimiza carteira de ETFs usando Mean-Variance Optimization
        
        Args:
            etf_data: Lista de objetos ETFData com informações dos ETFs
            target_return: Retorno alvo anual em percentual (opcional)
            risk_profile: Perfil de risco ("Conservador", "Moderado", "Agressivo", "Muito agressivo")
            constraints: Restrições adicionais para a otimização
            max_volatility: Volatilidade máxima permitida em percentual
            max_drawdown: Drawdown máximo permitido em percentual (valor positivo, será convertido para negativo)
            
        Returns:
            Objeto CarteiraOtimizada com a alocação ótima
        """
        await self._ensure_fmp_client()
        
        # Extrair dados para o cálculo
        symbols = [etf.symbol for etf in etf_data]
        returns = np.array([etf.retorno_anual for etf in etf_data]) / 100.0  # Converter para decimal
        
        # Obter matriz de correlação
        correlation_matrix = await self.fmp_client.calc_correlation_matrix(symbols)
        
        # Criar matriz de covariância
        volatilities = np.array([etf.volatilidade for etf in etf_data]) / 100.0  # Converter para decimal
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix.values
        
        # Aplicar restrições conforme perfil de risco
        if constraints is None:
            constraints = {}
        
        # Restrições baseadas no perfil de risco
        risk_constraints = self._get_risk_profile_constraints(risk_profile, etf_data)
        constraints = {**risk_constraints, **constraints}
        
        # Converter target_return para decimal se fornecido
        decimal_target_return = None
        if target_return is not None:
            decimal_target_return = target_return / 100.0

        # Converter max_volatility para decimal se fornecido
        if max_volatility is not None:
            constraints["max_volatility"] = max_volatility / 100.0
            
        # Verificar se o retorno alvo é viável
        if decimal_target_return is not None:
            # Verificar se o retorno alvo está dentro do intervalo possível
            max_possible_return = max(returns)
            min_possible_return = min(returns)
            
            if decimal_target_return > max_possible_return:
                # Se o retorno desejado for maior que o máximo possível, ajustar para o máximo e avisar
                decimal_target_return = max_possible_return
                
            if decimal_target_return < min_possible_return:
                # Se for menor que o mínimo possível, usar o mínimo
                decimal_target_return = min_possible_return
        
        # Realizar otimização
        if decimal_target_return is not None:
            # Caso 1: Otimizar para um retorno alvo específico (minimizando a volatilidade)
            portfolio = self._optimize_for_target_return(returns, cov_matrix, decimal_target_return, constraints)
        else:
            # Caso 2: Maximizar o índice Sharpe (retorno ajustado ao risco)
            portfolio = self._optimize_sharpe_ratio(returns, cov_matrix, constraints)
        
        # Calcular métricas da carteira otimizada
        weights = portfolio["weights"]
        portfolio_return = np.sum(returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        # Estimar drawdown máximo (aproximação baseada na volatilidade)
        max_drawdown_estimate = -2.5 * portfolio_volatility
        
        # Verificar se estamos respeitando a restrição de max_drawdown do usuário
        if max_drawdown is not None:
            # Converter para decimal e para valor negativo (drawdown é negativo)
            user_max_drawdown = -(max_drawdown / 100.0)
            
            # Se o drawdown estimado for pior (mais negativo) que o limite do usuário,
            # precisamos reotimizar com uma volatilidade máxima correspondente
            if max_drawdown_estimate < user_max_drawdown:
                # Calcular a volatilidade máxima que atende o max_drawdown
                derived_max_volatility = -user_max_drawdown / 2.5
                constraints["max_volatility"] = derived_max_volatility
                
                # Reotimizar com a nova restrição
                if decimal_target_return is not None:
                    portfolio = self._optimize_for_target_return(returns, cov_matrix, decimal_target_return, constraints)
                else:
                    portfolio = self._optimize_sharpe_ratio(returns, cov_matrix, constraints)
                    
                # Recalcular métricas
                weights = portfolio["weights"]
                portfolio_return = np.sum(returns * weights)
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
                max_drawdown_estimate = -2.5 * portfolio_volatility
        
        # Formatar resultados
        formatted_weights = [round(weight * 100, 2) for weight in weights]  # Converter para percentual
        
        # Criar lista de ETFs com seus dados e pesos
        etfs_list = []
        for i, etf in enumerate(etf_data):
            etfs_list.append({
                "symbol": etf.symbol,
                "name": etf.name,
                "categoria": etf.categoria,
                "expense_ratio": etf.expense_ratio,
                "peso": formatted_weights[i],
                "retorno_esperado": etf.retorno_anual,
                "volatilidade": etf.volatilidade
            })
        
        # Criar e retornar objeto CarteiraOtimizada
        return CarteiraOtimizada(
            etfs=etfs_list,
            pesos=formatted_weights,
            retorno_esperado=round(portfolio_return * 100, 2),  # Converter para percentual
            volatilidade=round(portfolio_volatility * 100, 2),  # Converter para percentual
            sharpe_ratio=round(sharpe_ratio, 2),
            max_drawdown=round(max_drawdown_estimate * 100, 2),  # Converter para percentual
            correlacao_matriz=correlation_matrix
        )
    
    def _get_risk_profile_constraints(self, risk_profile: str, etf_data: List[ETFData]) -> Dict[str, Any]:
        """
        Define restrições baseadas no perfil de risco
        
        Args:
            risk_profile: Perfil de risco do investidor
            etf_data: Lista de ETFs para otimização
            
        Returns:
            Dicionário com restrições para a otimização
        """
        # Mapear ETFs por categoria
        categorias = {}
        for i, etf in enumerate(etf_data):
            if etf.categoria not in categorias:
                categorias[etf.categoria] = []
            categorias[etf.categoria].append(i)
        
        # Definir restrições por perfil de risco
        constraints = {"min_weight": 0.01}  # Mínimo 1% por ETF
        
        if risk_profile == "Conservador":
            constraints["max_renda_variavel"] = 0.30
            constraints["min_renda_fixa"] = 0.50
            constraints["max_commodities"] = 0.10
            constraints["max_volatility"] = 0.10  # 10% máximo de volatilidade para conservador
        
        elif risk_profile == "Moderado":
            constraints["max_renda_variavel"] = 0.60
            constraints["min_renda_fixa"] = 0.30
            constraints["max_commodities"] = 0.15
            constraints["max_volatility"] = 0.15  # 15% máximo de volatilidade para moderado
        
        elif risk_profile == "Agressivo":
            constraints["max_renda_variavel"] = 0.80
            constraints["min_renda_fixa"] = 0.10
            constraints["max_commodities"] = 0.20
            constraints["max_volatility"] = 0.20  # 20% máximo de volatilidade para agressivo
        
        elif risk_profile == "Muito agressivo":
            constraints["max_renda_variavel"] = 0.90
            constraints["min_renda_fixa"] = 0.0
            constraints["max_commodities"] = 0.25
            constraints["max_volatility"] = 0.25  # 25% máximo de volatilidade para muito agressivo
        
        return constraints
    
    def _optimize_for_target_return(
        self, 
        returns: np.ndarray, 
        cov_matrix: np.ndarray, 
        target_return: float,
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Otimiza carteira para um retorno alvo específico
        
        Args:
            returns: Vetor de retornos esperados
            cov_matrix: Matriz de covariância
            target_return: Retorno alvo
            constraints: Restrições adicionais
            
        Returns:
            Dict com os pesos otimizados e outras informações
        """
        n = len(returns)
        
        # Função objetivo: minimizar a volatilidade
        def objective(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Restrições padrão
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Soma dos pesos = 1
            {'type': 'eq', 'fun': lambda x: np.dot(x, returns) - target_return}  # Retorno alvo
        ]
        
        # Adicionar restrição de volatilidade máxima, se especificada
        if "max_volatility" in constraints:
            max_volatility = constraints["max_volatility"]
            constraints_list.append({
                'type': 'ineq', 
                'fun': lambda x: max_volatility - np.sqrt(np.dot(x.T, np.dot(cov_matrix, x)))
            })
            
        # Adicionar restrições de categoria - vamos utilizar um conjunto de categorias conhecidas
        # em vez de tentar acessar etf_data que não está disponível neste contexto
        
        # Identificar índices por categoria através da análise de retornos/volatilidade
        # Simplificação: Ativos com maior volatilidade são considerados renda variável
        # Ativos com menor volatilidade são considerados renda fixa
        volatilidades = np.diag(cov_matrix)
        
        # Estimar índices de renda variável (30% mais voláteis)
        renda_variavel_indices = np.argsort(volatilidades)[-int(n*0.3):]
                
        if renda_variavel_indices.size > 0 and "max_renda_variavel" in constraints:
            max_rv = constraints["max_renda_variavel"]
            constraints_list.append({
                'type': 'ineq',
                'fun': lambda x: max_rv - sum(x[i] for i in renda_variavel_indices)
            })
        
        # Estimar índices de renda fixa (50% menos voláteis)
        renda_fixa_indices = np.argsort(volatilidades)[:int(n*0.5)]
                
        if renda_fixa_indices.size > 0 and "min_renda_fixa" in constraints:
            min_rf = constraints["min_renda_fixa"]
            constraints_list.append({
                'type': 'ineq',
                'fun': lambda x: sum(x[i] for i in renda_fixa_indices) - min_rf
            })
        
        # Commodities são mais difíceis de identificar sem informação contextual
        # Poderíamos usar uma abordagem mais sofisticada em uma implementação real
            
        # Limites por ETF - agora usando min_weight de 0 para permitir que o otimizador seja mais flexível
        # As restrições de alocação mínima serão aplicadas após a seleção dos 10 ETFs
        min_weight = 0.0  # Permitir pesos zero inicialmente
        max_weight = 0.25  # Máximo 25% em um único ativo
        bounds = [(min_weight, max_weight) for _ in range(n)]
        
        # Solução inicial equilibrada
        initial_weights = np.ones(n) / n
        
        try:
            # Resolver a otimização
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': 1000}
            )
            
            # Checar se a otimização foi bem-sucedida
            if result.success:
                # Normalizar pesos para garantir soma = 1
                weights = result.x / np.sum(result.x)
                portfolio_return = np.dot(weights, returns)
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                return {
                    "weights": weights,
                    "volatility": portfolio_volatility,
                    "return": portfolio_return
                }
                
        except Exception as e:
            print(f"Erro na otimização: {e}")
        
        # Fallback para pesos ajustados manualmente se a otimização falhar
        # Tenta encontrar uma combinação que satisfaça o retorno alvo
        # Começa priorizando ativos de menor risco
        sorted_indices = np.argsort(returns / np.sqrt(np.diag(cov_matrix)))[::-1]  # Melhor Sharpe primeiro
        fallback_weights = np.zeros(n)
        
        # Distribuir pesos iniciais
        remaining_weight = 1.0
        for i in sorted_indices:
            if remaining_weight <= 0:
                break
            fallback_weights[i] = min(0.25, remaining_weight)  # No máximo 25% em um único ativo
            remaining_weight -= fallback_weights[i]
            
        # Normalizar pesos
        fallback_weights = fallback_weights / np.sum(fallback_weights)
        
        # Calcular métricas do fallback
        fallback_return = np.dot(fallback_weights, returns)
        fallback_volatility = np.sqrt(np.dot(fallback_weights.T, np.dot(cov_matrix, fallback_weights)))
        
        return {
            "weights": fallback_weights,
            "volatility": fallback_volatility,
            "return": fallback_return
        }
    
    def _optimize_sharpe_ratio(
        self, 
        returns: np.ndarray, 
        cov_matrix: np.ndarray,
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Otimiza carteira para maximizar o índice Sharpe
        
        Args:
            returns: Vetor de retornos esperados
            cov_matrix: Matriz de covariância
            constraints: Restrições adicionais
            
        Returns:
            Dict com os pesos otimizados e outras informações
        """
        n = len(returns)
        
        # Função objetivo: minimizar o negativo do índice Sharpe
        def objective(weights):
            portfolio_return = np.dot(weights, returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        # Restrições
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Soma dos pesos = 1
        ]
        
        # Adicionar restrição de volatilidade máxima, se especificada
        if "max_volatility" in constraints:
            max_volatility = constraints["max_volatility"]
            constraints_list.append({
                'type': 'ineq', 
                'fun': lambda x: max_volatility - np.sqrt(np.dot(x.T, np.dot(cov_matrix, x)))
            })
            
        # Adicionar restrições de categoria usando a mesma abordagem da função _optimize_for_target_return
        # Identificar índices por categoria através da análise de retornos/volatilidade
        volatilidades = np.diag(cov_matrix)
        
        # Estimar índices de renda variável (30% mais voláteis)
        renda_variavel_indices = np.argsort(volatilidades)[-int(n*0.3):]
                
        if renda_variavel_indices.size > 0 and "max_renda_variavel" in constraints:
            max_rv = constraints["max_renda_variavel"]
            constraints_list.append({
                'type': 'ineq',
                'fun': lambda x: max_rv - sum(x[i] for i in renda_variavel_indices)
            })
        
        # Estimar índices de renda fixa (50% menos voláteis)
        renda_fixa_indices = np.argsort(volatilidades)[:int(n*0.5)]
                
        if renda_fixa_indices.size > 0 and "min_renda_fixa" in constraints:
            min_rf = constraints["min_renda_fixa"]
            constraints_list.append({
                'type': 'ineq',
                'fun': lambda x: sum(x[i] for i in renda_fixa_indices) - min_rf
            })
            
        # Limites por ETF
        min_weight = 0.0  # Permitir pesos zero inicialmente
        max_weight = 0.25  # Máximo 25% em um único ativo
        bounds = [(min_weight, max_weight) for _ in range(n)]
        
        # Solução inicial equilibrada
        initial_weights = np.ones(n) / n
        
        try:
            # Resolver a otimização
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': 1000}
            )
            
            # Checar se a otimização foi bem-sucedida
            if result.success:
                # Normalizar pesos para garantir soma = 1
                weights = result.x / np.sum(result.x)
                portfolio_return = np.dot(weights, returns)
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                return {
                    "weights": weights,
                    "volatility": portfolio_volatility,
                    "return": portfolio_return
                }
                
        except Exception as e:
            print(f"Erro na otimização: {e}")
        
        # Fallback para pesos iguais se a otimização falhar
        equal_weights = np.ones(n) / n
        portfolio_return = np.dot(equal_weights, returns)
        portfolio_volatility = np.sqrt(np.dot(equal_weights.T, np.dot(cov_matrix, equal_weights)))
        
        return {
            "weights": equal_weights,
            "volatility": portfolio_volatility,
            "return": portfolio_return
        } 