import streamlit as st

def show():
    """
    Renderiza a página inicial do aplicativo com design aprimorado
    """
    # CSS personalizado para melhorar a aparência
    st.markdown("""
    <style>
        .main-title {
            font-size: 2.5rem !important;
            font-weight: 600 !important;
            margin-bottom: 0.8rem !important;
            color: #121A3E;
        }
        .hero-container {
            background: linear-gradient(135deg, #4361EE 0%, #3A0CA3 100%);
            padding: 3rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            color: white;
            box-shadow: 0 10px 20px rgba(67, 97, 238, 0.15);
        }
        .hero-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 1rem;
            line-height: 1.2;
        }
        .hero-subtitle {
            font-size: 1.3rem;
            opacity: 0.9;
            margin-bottom: 2rem;
            line-height: 1.5;
        }
        .feature-card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.06);
            height: 100%;
            transition: transform 0.3s;
        }
        .feature-card:hover {
            transform: translateY(-5px);
        }
        .feature-icon {
            font-size: 2rem;
            margin-bottom: 1rem;
            color: #4361EE;
        }
        .feature-title {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: #121A3E;
        }
        .feature-description {
            color: #4F5665;
            font-size: 1rem;
            line-height: 1.5;
        }
        .cta-section {
            background-color: #F8F9FE;
            padding: 2rem;
            border-radius: 12px;
            text-align: center;
            margin-top: 2rem;
            margin-bottom: 2rem;
        }
        .steps-container {
            background-color: white;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.06);
            margin-bottom: 2rem;
        }
        .step-number {
            background-color: #4361EE;
            color: white;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            margin-right: 10px;
            font-weight: bold;
        }
        .step-text {
            display: inline;
            font-size: 1.1rem;
            vertical-align: middle;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Hero section
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">ETF Blueprint</div>
        <div class="hero-subtitle">Sua carteira global de ETFs otimizada por dados e algoritmos avançados, pronta em minutos.</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Como funciona
    st.markdown('<div class="steps-container">', unsafe_allow_html=True)
    st.subheader("Como funciona")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div style="margin-bottom: 15px;">
            <div class="step-number">1</div>
            <div class="step-text">Preencha seu perfil de investimento e objetivos</div>
        </div>
        <div style="margin-bottom: 15px;">
            <div class="step-number">2</div>
            <div class="step-text">Nossos algoritmos otimizam sua carteira personalizada</div>
        </div>
        <div style="margin-bottom: 15px;">
            <div class="step-number">3</div>
            <div class="step-text">Receba uma análise detalhada com visualizações claras</div>
        </div>
        <div style="margin-bottom: 15px;">
            <div class="step-number">4</div>
            <div class="step-text">Exporte sua estratégia para implementação imediata</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        try:
            print("Tentando carregar imagem: assets/investment_chart.png")
            import os
            print(f"Diretório atual: {os.getcwd()}")
            if os.path.exists("assets/investment_chart.png"):
                print("Arquivo encontrado, carregando...")
                st.image("assets/investment_chart.png")
            else:
                print("Arquivo de imagem não encontrado, mostrando alternativa")
                st.info("Visualização de exemplo da carteira otimizada")
                
                # Criar gráfico com Plotly como alternativa
                import plotly.express as px
                import pandas as pd
                import numpy as np
                
                # Dados de exemplo
                etfs = ["VTI", "VEA", "VWO", "BND", "BNDX"]
                pesos = [40, 20, 10, 20, 10]
                
                # Criar dataframe
                df = pd.DataFrame({"ETF": etfs, "Peso": pesos})
                
                # Criar gráfico de pizza
                fig = px.pie(df, values="Peso", names="ETF", title="Exemplo de Alocação")
                st.plotly_chart(fig)
        except Exception as e:
            print(f"Erro ao carregar imagem: {str(e)}")
            st.warning("Não foi possível carregar a visualização")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Botão de Call-to-Action
    st.markdown('<div class="cta-section">', unsafe_allow_html=True)
    st.markdown('<h2 style="margin-bottom: 20px; color: #121A3E;">Pronto para otimizar seus investimentos?</h2>', unsafe_allow_html=True)
    
    # Adicionando um botão principal e um link alternativo para garantir navegação
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("Começar agora", type="primary", key="btn_comecar"):
            # Definir diretamente a navegação sem mecanismos complexos
            st.session_state.nav = "Perfil de Risco"
            st.toast("Redirecionando para página de perfil...")
    
    with col2:
        # Link alternativo como backup para garantir navegação
        st.markdown('<a href="?nav=perfil" target="_self" style="text-decoration: none;"><button style="background-color: #4361EE; color: white; border: none; padding: 10px 15px; border-radius: 5px; cursor: pointer;">Ir para perfil</button></a>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Recursos em cards
    st.subheader("Por que usar o ETF Blueprint?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Dados em tempo real</div>
            <div class="feature-description">
                Utilizamos dados financeiros atualizados de mercado para otimizar sua carteira com precisão.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <div class="feature-title">Análise personalizada</div>
            <div class="feature-description">
                Cada carteira é adaptada ao seu perfil de risco, horizonte de investimento e objetivos financeiros.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📝</div>
            <div class="feature-title">Explicação simplificada</div>
            <div class="feature-description">
                Receba uma narrativa em linguagem simples que explica sua estratégia e facilita a compreensão.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer com informações adicionais
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; color: #666; font-size: 0.9rem;">
        ETF Blueprint utiliza a API Financial Modeling Prep para dados de mercado e tecnologias de IA para análises personalizadas.
    </div>
    """, unsafe_allow_html=True) 