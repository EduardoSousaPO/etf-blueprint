import os
import json
import aiohttp
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
import asyncio

class NarratorService:
    """
    Serviço para gerar narrativas explicativas sobre carteiras de ETFs
    utilizando a API da OpenAI
    """
    
    # URL base da API
    API_URL = "https://api.openai.com/v1/chat/completions"
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        """
        Inicializa o serviço de narrativa
        
        Args:
            api_key: Chave de API OpenAI. Se não fornecida, tenta ler de OPENAI_API_KEY
            model: Modelo da OpenAI a ser utilizado (default: gpt-4o)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API Key não encontrada. Defina OPENAI_API_KEY ou passe como parâmetro.")
        
        print("NarratorService inicializado")
        
        self.model = model
    
    async def generate_narrative(
        self,
        perfil: Dict[str, Any],
        carteira: Dict[str, Any],
        metricas: Dict[str, float],
        max_palavras: int = 400
    ) -> str:
        """
        Gera uma narrativa explicativa sobre a carteira de ETFs
        
        Args:
            perfil: Dicionário com dados do perfil do investidor
            carteira: Dicionário com informações sobre a carteira
            metricas: Dicionário com métricas da carteira (retorno, volatilidade, etc.)
            max_palavras: Número máximo de palavras da narrativa
            
        Returns:
            Texto narrativo em linguagem natural
        """
        # Construir o prompt
        prompt = self._build_narrative_prompt(perfil, carteira, metricas, max_palavras)
        
        # Enviar para a API
        response = await self._call_openai_api(prompt)
        
        # Extrair e retornar a narrativa
        narrative = response["choices"][0]["message"]["content"].strip()
        
        # Validar se a narrativa não está vazia
        if not narrative:
            return self._generate_fallback_narrative(perfil, carteira, metricas)
        
        return narrative
    
    def _build_narrative_prompt(
        self,
        perfil: Dict[str, Any],
        carteira: Dict[str, Any],
        metricas: Dict[str, float],
        max_palavras: int
    ) -> List[Dict[str, str]]:
        """
        Constrói o prompt para enviar à API OpenAI
        
        Args:
            perfil: Dados do perfil do investidor
            carteira: Dados da carteira otimizada
            metricas: Métricas da carteira
            max_palavras: Número máximo de palavras
            
        Returns:
            Lista de mensagens formatadas para a API
        """
        # Formatação dos ETFs e seus pesos
        etfs_info = ""
        for etf in carteira["etfs"]:
            etfs_info += f"- {etf['symbol']} ({etf['name']}): {etf['peso']}%, categoria: {etf['categoria']}\n"
        
        # Prompt do sistema
        system_prompt = """
        Você é um consultor financeiro especializado em ETFs (Exchange Traded Funds) globais.
        Sua tarefa é explicar de forma clara e educativa uma carteira de investimentos em ETFs 
        para um cliente, considerando seu perfil de risco e objetivos.
        
        Use linguagem simples e direta, evitando jargões financeiros complexos.
        Foque em explicar o raciocínio por trás da alocação e como a carteira atende ao perfil do cliente.
        Não use linguagem promocional ou de vendas.
        """
        
        # Prompt do usuário
        user_prompt = f"""
        Por favor, crie uma narrativa explicativa sobre a carteira de ETFs abaixo.
        A narrativa deve ter no máximo {max_palavras} palavras e deve ser escrita em português.
        
        PERFIL DO INVESTIDOR:
        - Nome: {perfil.get('nome', 'Cliente')}
        - Horizonte de investimento: {perfil.get('horizonte', 'N/A')} anos
        - Perfil de risco: {perfil.get('perfil_risco', 'N/A')}
        - Tolerância a drawdown: {perfil.get('tolerancia_drawdown', 'N/A')}
        - Retorno alvo: {perfil.get('retorno_alvo', 'N/A')}%
        - Objetivos: {', '.join(perfil.get('objetivos', ['N/A']))}
        - Regiões preferidas: {', '.join(perfil.get('regioes', ['Global']))}
        
        CARTEIRA RECOMENDADA:
        {etfs_info}
        
        MÉTRICAS DA CARTEIRA:
        - Retorno esperado: {metricas.get('retorno_esperado', 'N/A')}%
        - Volatilidade: {metricas.get('volatilidade', 'N/A')}%
        - Índice Sharpe: {metricas.get('sharpe_ratio', 'N/A')}
        - Drawdown máximo estimado: {metricas.get('max_drawdown', 'N/A')}%
        
        A narrativa deve:
        1. Explicar como a carteira se alinha ao perfil e objetivos do investidor
        2. Destacar o papel de cada ETF principal na carteira (top 3 em alocação)
        3. Explicar o equilíbrio entre risco e retorno da carteira
        4. Fornecer breves recomendações sobre rebalanceamento e monitoramento
        5. Abordar como a carteira se comportaria em diferentes cenários de mercado
        
        Evite repetir os dados acima de forma mecânica. Elabore uma narrativa fluida e educativa.
        """
        
        # Estrutura de mensagens para API
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return messages
    
    async def _call_openai_api(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Realiza chamada à API da OpenAI
        
        Args:
            messages: Lista de mensagens para o modelo
            
        Returns:
            Resposta da API como dict
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,  # Baixa temperatura para respostas mais consistentes
            "max_tokens": 800,  # Aproximadamente 600 palavras
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.API_URL, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Erro na API OpenAI: {response.status} - {error_text}")
                
                return await response.json()
    
    def _generate_fallback_narrative(
        self,
        perfil: Dict[str, Any],
        carteira: Dict[str, Any],
        metricas: Dict[str, float]
    ) -> str:
        """
        Gera uma narrativa padrão caso a API falhe
        
        Args:
            perfil: Dados do perfil do investidor
            carteira: Dados da carteira otimizada
            metricas: Métricas da carteira
            
        Returns:
            Texto narrativo padrão
        """
        # Obter 3 principais ETFs
        top_etfs = sorted(carteira["etfs"], key=lambda x: x["peso"], reverse=True)[:3]
        top_etfs_str = ", ".join([f"{etf['symbol']} ({etf['peso']}%)" for etf in top_etfs])
        
        return f"""
        Com base no seu perfil {perfil.get('perfil_risco', 'de investidor')} e horizonte de {perfil.get('horizonte', 'longo')} anos, 
        desenvolvemos uma carteira diversificada globalmente com foco em crescimento sustentável e proteção.
        
        Sua carteira tem um retorno esperado de {metricas.get('retorno_esperado', '8')}% ao ano, 
        com volatilidade estimada de {metricas.get('volatilidade', '12')}%. 
        O índice Sharpe, que mede o retorno ajustado ao risco, é de {metricas.get('sharpe_ratio', '0.7')}.
        
        Os principais ETFs da sua alocação são {top_etfs_str}, que oferecem uma 
        combinação equilibrada de exposição aos mercados globais com risco controlado.
        
        Recomendamos revisar esta alocação anualmente ou quando houver mudanças significativas 
        em seus objetivos financeiros. Para manter a estratégia, é importante rebalancear 
        periodicamente para manter os percentuais-alvo.
        
        Esta carteira está alinhada com seu objetivo de retorno de {perfil.get('retorno_alvo', '8')}% 
        e foi desenhada para suportar períodos de volatilidade mercado, mantendo um perfil de 
        risco adequado às suas necessidades.
        """

    async def generate_portfolio_narrative(self, params: Dict[str, Any]) -> str:
        """
        Gera uma narrativa explicativa sobre a carteira otimizada
        
        Args:
            params: Dicionário com parâmetros necessários para a narrativa
                - perfil: Dados do perfil do investidor
                - carteira: Objeto CarteiraOtimizada 
                - horizonte: Horizonte de investimento em anos
                - objetivo: Objetivo de retorno anual
                
        Returns:
            Texto narrativo em linguagem natural
        """
        try:
            # Extrair parâmetros
            perfil = params.get("perfil", {})
            carteira_obj = params.get("carteira", None)
            
            if not carteira_obj:
                return "Não foi possível gerar a narrativa: dados da carteira ausentes."
            
            # Converter objeto CarteiraOtimizada para dicionário
            carteira = {
                "etfs": carteira_obj.etfs,
                "pesos": carteira_obj.pesos,
                "retorno_esperado": carteira_obj.retorno_esperado,
                "volatilidade": carteira_obj.volatilidade,
                "sharpe_ratio": carteira_obj.sharpe_ratio,
                "max_drawdown": carteira_obj.max_drawdown
            }
            
            # Preparar métricas
            metricas = {
                "retorno_esperado": carteira_obj.retorno_esperado,
                "volatilidade": carteira_obj.volatilidade,
                "sharpe_ratio": carteira_obj.sharpe_ratio,
                "max_drawdown": carteira_obj.max_drawdown
            }
            
            # Gerar a narrativa usando a função existente
            return await self.generate_narrative(perfil, carteira, metricas)
            
        except Exception as e:
            # Em caso de erro, retornar uma mensagem genérica
            print(f"Erro ao gerar narrativa: {str(e)}")
            return f"""
            Com base no seu perfil {perfil.get('perfil_risco', 'de investidor')} e objetivo de retorno de {perfil.get('retorno_alvo', '8')}%,
            desenvolvemos uma carteira diversificada de ETFs globais.
            
            Esta carteira foi otimizada para equilibrar risco e retorno, considerando seu horizonte de investimento
            e tolerância a quedas de mercado.
            
            Recomendamos revisar esta carteira anualmente ou quando houver mudanças significativas em seus objetivos financeiros.
            """ 