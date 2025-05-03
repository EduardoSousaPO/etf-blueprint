import streamlit as st

def show():
    """
    Renderiza a página inicial do aplicativo
    """
    st.title("Bem-vindo ao ETF Blueprint")
    
    # Frase mantra em destaque
    st.markdown(
        """
        <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;'>
            <h2 style='color: #1e3a8a; font-weight: bold;'>Os dados vão salvar seus investimentos.</h2>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    # Descrição do serviço
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Sua carteira global de ETFs em apenas 5 minutos")
        st.markdown(
            """
            O ETF Blueprint utiliza algoritmos avançados para criar uma carteira de ETFs 
            personalizada de acordo com seu perfil de risco e objetivos financeiros.
            
            **Como funciona:**
            1. Preencha seu perfil de investimento
            2. Nossos algoritmos otimizam sua carteira
            3. Receba um relatório completo em PDF
            4. Implemente sua estratégia com o CSV exportado
            """
        )
        
        # Botão com implementação alternativa
        btn_comecar = st.button("Começar agora", type="primary", use_container_width=True, key="btn_comecar")
        if btn_comecar:
            # Definir estado de navegação
            st.session_state["nav"] = "Perfil de Risco"
            # Forçar recarregamento da página
            st.query_params["page"] = "perfil"
            st.rerun()
    
    with col2:
        st.image("assets/investment_chart.png", use_column_width=True)
    
    # Vantagens
    st.markdown("---")
    st.subheader("Por que usar o ETF Blueprint?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 Dados em tempo real")
        st.markdown("Utilizamos dados financeiros atualizados para otimizar sua carteira.")
    
    with col2:
        st.markdown("### 🔍 Análise personalizada")
        st.markdown("Cada carteira é adaptada ao seu perfil de risco e objetivos.")
    
    with col3:
        st.markdown("### 📝 Explicação simplificada")
        st.markdown("Receba uma narrativa em linguagem simples sobre sua estratégia.") 