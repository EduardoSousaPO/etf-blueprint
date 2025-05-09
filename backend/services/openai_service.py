import os
import streamlit as st
import openai

# Obter chave API da OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))

# Configurar a API da OpenAI
openai.api_key = OPENAI_API_KEY

def gerar_texto(carteira, perfil):
    """
    Gera análise de texto para uma carteira de ETFs usando a API OpenAI.
    
    Args:
        carteira (dict): Dicionário com alocações de ETFs.
        perfil (str): Perfil do investidor ('Conservador', 'Moderado', 'Agressivo').
        
    Returns:
        str: Texto da análise gerada.
    """
    # Verificar se a chave API está configurada
    if not OPENAI_API_KEY:
        return "Chave API da OpenAI não configurada. Configure a chave em .env ou Secrets do Streamlit."
    
    # Construir o prompt para a API
    prompt = f"""
    Você é um analista financeiro especializado em ETFs. 
    Explique esta carteira de 10 ETFs: {carteira}
    para um investidor com perfil {perfil}, em até 400 palavras, em português simples.
    
    A explicação deve incluir:
    1. Uma visão geral da carteira e como ela se alinha ao perfil do investidor
    2. Uma breve descrição dos principais ETFs na carteira
    3. Os potenciais riscos e benefícios desta alocação
    4. Algumas considerações sobre o horizonte de investimento recomendado
    
    Use linguagem clara e objetiva, adequada para um investidor de varejo.
    """
    
    try:
        # Realizar a chamada à API
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2  # Valor baixo para respostas mais consistentes
        )
        
        # Extrair e retornar o texto da resposta
        return resp.choices[0].message.content.strip()
    
    except Exception as e:
        # Em caso de erro, retornar uma mensagem padrão
        return f"Não foi possível gerar a análise: {str(e)}" 