import streamlit as st
import pandas as pd
from datetime import datetime
import json
import os
import asyncio

def show():
    """
    Renderiza a página de perfil de risco do investidor
    """
    # Estilos personalizados
    st.markdown("""
    <style>
        /* Estilo geral */
        .profile-page {
            padding: 1rem 0;
        }
        
        /* Card de formulário */
        .form-card {
            background-color: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.06);
            margin-bottom: 20px;
        }
        
        /* Títulos de seção */
        .section-title {
            position: relative;
            padding-bottom: 10px;
            margin-bottom: 20px;
            font-weight: 600;
            color: #121A3E;
        }
        .section-title:after {
            content: "";
            position: absolute;
            left: 0;
            bottom: 0;
            width: 50px;
            height: 3px;
            background-color: #4361EE;
        }
        
        /* Progress bar */
        .stProgress .st-bo {
            background-color: #4361EE !important;
        }
        
        /* Destaque para objetivos */
        .goal-selector {
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            transition: all 0.3s;
            border: 2px solid transparent;
        }
        .goal-selector:hover {
            border-color: #e0e0ef;
            background-color: #f8f9fe;
        }
        .goal-title {
            font-weight: 600;
            color: #121A3E;
            margin-bottom: 5px;
        }
        .goal-description {
            font-size: 0.9rem;
            color: #4F5665;
        }
        
        /* Range slider personalizado */
        .custom-slider {
            background-color: #f8f9fe;
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0;
        }
        .slider-label {
            display: flex;
            justify-content: space-between;
            margin-top: 10px;
        }
        .slider-min, .slider-max {
            font-size: 0.85rem;
            color: #4F5665;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="profile-page">', unsafe_allow_html=True)
    st.title("Seu Perfil de Investimento")
    
    # Formulário para capturar dados do perfil
    with st.form("form_perfil"):
        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-title">Objetivos Financeiros</h2>', unsafe_allow_html=True)
        
        # Horizonte de investimento
        st.markdown("**Qual é o seu horizonte de investimento?**")
        horizonte = st.radio(
            label="Horizonte",
            options=["Curto prazo (1-2 anos)", "Médio prazo (3-5 anos)", "Longo prazo (6+ anos)"],
            label_visibility="collapsed"
        )
        
        # Objetivo principal
        st.markdown("**Qual é o seu objetivo principal?**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="goal-selector">
                <div class="goal-title">Preservação de Capital</div>
                <div class="goal-description">Proteger o patrimônio, com foco em segurança e baixa volatilidade</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="goal-selector">
                <div class="goal-title">Crescimento Moderado</div>
                <div class="goal-description">Crescimento equilibrado, com riscos controlados</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="goal-selector">
                <div class="goal-title">Crescimento Agressivo</div>
                <div class="goal-description">Maximizar retornos, aceitando maior volatilidade</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="goal-selector">
                <div class="goal-title">Renda Passiva</div>
                <div class="goal-description">Geração de fluxo de caixa regular através de dividendos</div>
            </div>
            """, unsafe_allow_html=True)
        
        objetivo = st.radio(
            label="Objetivo",
            options=["Preservação de Capital", "Crescimento Moderado", "Crescimento Agressivo", "Renda Passiva"],
            label_visibility="collapsed"
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Perfil de risco
        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-title">Perfil de Risco</h2>', unsafe_allow_html=True)
        
        st.markdown("**Como você reage quando seus investimentos caem significativamente?**")
        reacao_perda = st.radio(
            label="Reação a perdas",
            options=[
                "Vendo imediatamente para evitar mais perdas",
                "Fico preocupado, mas aguardo um pouco",
                "Mantenho a calma, pois faz parte do ciclo de investimentos",
                "Aproveito para comprar mais, pois os preços estão mais baixos"
            ]
        )
        
        st.markdown("**Qual é o seu nível de experiência com investimentos?**")
        experiencia = st.radio(
            label="Experiência",
            options=[
                "Iniciante - Estou começando a investir agora",
                "Intermediário - Já invisto há algum tempo",
                "Avançado - Tenho bastante experiência com diversos ativos"
            ]
        )
        
        # Tolerância a drawdown com slider customizado
        st.markdown("**Qual queda máxima (drawdown) você estaria disposto a tolerar temporariamente?**")
        st.markdown('<div class="custom-slider">', unsafe_allow_html=True)
        tolerancia_drawdown = st.slider(
            label="Tolerância a drawdown",
            min_value=5,
            max_value=40,
            value=20,
            step=5,
            format="%d%%",
            label_visibility="collapsed"
        )
        st.markdown("""
        <div class="slider-label">
            <div class="slider-min">Conservador (-5%)</div>
            <div class="slider-max">Arrojado (-40%)</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Retorno alvo com slider customizado
        st.markdown("**Qual retorno anual você espera alcançar?**")
        st.markdown('<div class="custom-slider">', unsafe_allow_html=True)
        retorno_alvo = st.slider(
            label="Retorno alvo",
            min_value=6,
            max_value=25,
            value=15,
            step=1,
            format="%d%%",
            label_visibility="collapsed"
        )
        st.markdown("""
        <div class="slider-label">
            <div class="slider-min">Conservador (6%)</div>
            <div class="slider-max">Arrojado (25%)</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Determinar perfil de risco automático com base nas respostas
        pontos_risco = 0
        
        # Horizonte
        if horizonte == "Curto prazo (1-2 anos)":
            pontos_risco += 1
        elif horizonte == "Médio prazo (3-5 anos)":
            pontos_risco += 2
        else:  # Longo prazo
            pontos_risco += 3
            
        # Objetivo
        if objetivo == "Preservação de Capital":
            pontos_risco += 1
        elif objetivo == "Renda Passiva" or objetivo == "Crescimento Moderado":
            pontos_risco += 2
        else:  # Crescimento agressivo
            pontos_risco += 3
            
        # Reação a perdas
        if reacao_perda == "Vendo imediatamente para evitar mais perdas":
            pontos_risco += 1
        elif reacao_perda == "Fico preocupado, mas aguardo um pouco":
            pontos_risco += 2
        elif reacao_perda == "Mantenho a calma, pois faz parte do ciclo de investimentos":
            pontos_risco += 3
        else:  # Compra mais
            pontos_risco += 4
            
        # Experiência
        if experiencia == "Iniciante - Estou começando a investir agora":
            pontos_risco += 1
        elif experiencia == "Intermediário - Já invisto há algum tempo":
            pontos_risco += 2
        else:  # Avançado
            pontos_risco += 3
            
        # Tolerância a drawdown (convertida proporcionalmente de 1-4)
        pontos_tolerancia = 1 + (tolerancia_drawdown - 5) / (40 - 5) * 3
        pontos_risco += pontos_tolerancia
        
        # Determinar perfil com base na pontuação total
        pontuacao_max = 3 + 3 + 4 + 3 + 4  # 17 pontos max
        porcentagem_risco = pontos_risco / pontuacao_max
        
        if porcentagem_risco < 0.4:
            perfil_risco = "Conservador"
        elif porcentagem_risco < 0.7:
            perfil_risco = "Moderado"
        else:
            perfil_risco = "Arrojado"
        
        # Converter tolerância em valor numérico
        tolerancia_drawdown_numeric = -tolerancia_drawdown  # valor negativo para refletir perda
        
        # Botão para submeter formulário
        st.markdown("</div>", unsafe_allow_html=True)
        
        submit_button = st.form_submit_button(
            label="Otimizar minha carteira",
            type="primary",
            use_container_width=True
        )
    
    # Se o botão foi clicado
    if submit_button:
        with st.spinner("Processando seu perfil..."):
            # Criar dicionário de perfil
            perfil = {
                "horizonte": horizonte,
                "objetivo": objetivo,
                "reacao_perda": reacao_perda,
                "experiencia": experiencia,
                "tolerancia_drawdown": tolerancia_drawdown,
                "tolerancia_drawdown_numeric": tolerancia_drawdown_numeric,
                "retorno_alvo": retorno_alvo,
                "perfil_risco": perfil_risco
            }
            
            # Armazenar no session_state
            st.session_state["perfil"] = perfil
            
            # Feedback ao usuário
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.success(f"Perfil de risco: {perfil_risco}")
                
                # Indicador de progresso
                st.progress(1.0)
                
                st.markdown(f"""
                <div style="background-color: #F0F7FF; border-radius: 10px; padding: 15px; margin-top: 20px; margin-bottom: 20px; text-align: center;">
                    <p style="margin-bottom: 10px;"><b>Seu perfil foi definido como {perfil_risco}</b></p>
                    <p style="font-size: 0.9rem; color: #4F5665;">Retorno alvo: {retorno_alvo}%</p>
                    <p style="font-size: 0.9rem; color: #4F5665;">Tolerância a drawdown: {tolerancia_drawdown}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.button(
                    "Ver minha carteira otimizada",
                    type="primary",
                    on_click=lambda: _navegar_para_resultados(),
                    use_container_width=True
                )
    
    st.markdown('</div>', unsafe_allow_html=True)

def _navegar_para_resultados():
    """
    Função auxiliar para navegação
    """
    st.session_state.nav = "Resultados"
    st.query_params["page"] = "resultados" 