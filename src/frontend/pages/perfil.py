import streamlit as st
import pandas as pd
from datetime import datetime
import json
import os

def show():
    """
    Renderiza a página de perfil de risco do usuário
    """
    st.title("Perfil de Risco")
    
    st.markdown("""
    Vamos entender seu perfil de investidor para criar uma carteira personalizada.
    Preencha as informações abaixo com cuidado para obter os melhores resultados.
    """)
    
    # Formulário de perfil
    with st.form("perfil_form"):
        # Informações básicas
        st.subheader("Informações básicas")
        col1, col2 = st.columns(2)
        
        with col1:
            nome = st.text_input("Nome", placeholder="Seu nome completo")
        
        with col2:
            email = st.text_input("Email", placeholder="seu@email.com")
        
        # Horizonte de investimento
        st.subheader("Horizonte de investimento")
        horizonte = st.slider(
            "Por quanto tempo você pretende manter seus investimentos?",
            min_value=1,
            max_value=30,
            value=5,
            step=1,
            help="Selecione o número de anos"
        )
        
        # Tolerância a risco
        st.subheader("Tolerância a risco")
        
        tolerancia_drawdown = st.select_slider(
            "Qual é a máxima queda (drawdown) que você aceitaria em sua carteira?",
            options=["5%", "10%", "15%", "20%", "25%", "30%", "35%", "40%"],
            value="15%",
            help="Drawdown é a queda percentual máxima do valor da carteira em relação ao seu pico anterior"
        )
        
        risco_opcoes = {
            "Conservador": "Prefiro estabilidade e baixa volatilidade, mesmo com retornos menores",
            "Moderado": "Busco equilíbrio entre risco e retorno",
            "Agressivo": "Aceito maior volatilidade em busca de retornos superiores",
            "Muito agressivo": "Foco em maximizar retornos de longo prazo, aceitando alta volatilidade"
        }
        
        perfil_risco = st.radio(
            "Como você descreveria seu perfil de risco?",
            options=list(risco_opcoes.keys()),
            index=1,
            help="Esta definição ajuda a determinar a alocação entre classes de ativos"
        )
        
        st.caption(risco_opcoes[perfil_risco])
        
        # Objetivos financeiros
        st.subheader("Objetivos financeiros")
        
        retorno_alvo = st.slider(
            "Qual é seu objetivo de retorno anual?",
            min_value=3.0,
            max_value=20.0,
            value=8.0,
            step=0.5,
            format="%.1f%%",
            help="O algoritmo tentará otimizar para este retorno, considerando seu perfil de risco"
        )
        
        objetivos = st.multiselect(
            "Quais são seus principais objetivos?",
            [
                "Aposentadoria", 
                "Independência financeira",
                "Compra de imóvel",
                "Educação",
                "Reserva de emergência",
                "Viagens",
                "Proteção patrimonial",
                "Crescimento patrimonial"
            ],
            default=["Crescimento patrimonial"],
            help="Selecione um ou mais objetivos"
        )
        
        # Preferências de investimento
        st.subheader("Preferências de investimento")
        
        regioes = st.multiselect(
            "Quais regiões você prefere incluir em sua carteira?",
            [
                "EUA",
                "Europa",
                "Ásia-Pacífico",
                "Mercados Emergentes",
                "Global (todas regiões)"
            ],
            default=["Global (todas regiões)"],
            help="Selecione as regiões geográficas para incluir na sua carteira"
        )
        
        # Botão de envio
        submitted = st.form_submit_button("Gerar minha carteira", use_container_width=True, type="primary")
        
        if submitted:
            if not nome or not email:
                st.error("Por favor, preencha nome e email antes de continuar.")
            else:
                # Salvar dados do perfil
                # Converter tolerância de drawdown para um valor numérico (remover o símbolo %)
                tolerancia_drawdown_numeric = float(tolerancia_drawdown.replace("%", ""))
                
                perfil_data = {
                    "nome": nome,
                    "email": email,
                    "horizonte": horizonte,
                    "tolerancia_drawdown": tolerancia_drawdown,
                    "tolerancia_drawdown_numeric": tolerancia_drawdown_numeric,  # Valor numérico para cálculos
                    "perfil_risco": perfil_risco,
                    "retorno_alvo": retorno_alvo,
                    "objetivos": objetivos,
                    "regioes": regioes,
                    "data_criacao": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Criar diretório temp se não existir
                os.makedirs("temp", exist_ok=True)
                
                # Salvar perfil temporariamente
                with open("temp/perfil.json", "w", encoding="utf-8") as f:
                    json.dump(perfil_data, f, ensure_ascii=False, indent=4)
                
                # Adicionar à sessão e redirecionar
                st.session_state["perfil"] = perfil_data
                st.session_state["show_results"] = True
                
                # Indicação visual de sucesso
                st.success("Perfil salvo com sucesso! Redirecionando para os resultados...")
                
                # Redirecionar para página de resultados usando o método atualizado
                st.session_state.nav = "Resultados"
                st.rerun() 