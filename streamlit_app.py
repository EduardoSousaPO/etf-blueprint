import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import requests
import json
import openai
from dotenv import load_dotenv
from datetime import datetime
import io
from PIL import Image
import matplotlib.pyplot as plt

# Configuração Inicial
st.set_page_config(
    page_title="ETF Blueprint",
    page_icon="📈",
    layout="wide"
)

# Esconder o menu do Streamlit e o rodapé
hide_menu_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display:none;}
    </style>
    """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# Carregar variáveis de ambiente
load_dotenv()

# Chaves API
FMP_API_KEY = os.getenv("FMP_API_KEY", st.secrets.get("FMP_API_KEY", ""))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))

# Configurar OpenAI
openai.api_key = OPENAI_API_KEY

# Definir listas de ETFs
ETFS_BR = [
    "BBSD11", "BOVA11", "BOVB11", "BOVS11", "BOVV11", "BRAX11", 
    "DIVO11", "ECOO11", "FIND11", "GOVE11", "HASH11", "ISPT11", 
    "IVVB11", "META11", "PIBB11", "QBTC11", "QETH11", "RICH11", 
    "SMAL11", "URA11", "XFIX11", "XINA11"
]

ETFS_EUA = [
    "SPY", "QQQ", "IWM", "DIA", "VTI", "IEFA", "EFA", "IEMG", "EEM", 
    "AGG", "BND", "LQD", "HYG", "TLT", "IEF", "GLD", "IAU", "SLV", 
    "XLK", "XLF", "XLV", "XLE", "XLY", "XLI", "XLP", "XLU", "XLB", 
    "XLC", "XLRE", "VGT", "VHT", "VOX", "VCR", "VAW", "VPU", "VFH", 
    "VDE", "VIS", "VDC", "VNQ", "IYR", "ARKK", "ARKW", "ARKF", "ARKG", 
    "ARKX", "QYLD", "JEPI", "JEPQ", "SCHD"
]

# Função para filtrar universo de ETFs
def filtrar_universo(opcao):
    if opcao == "BR":
        return ETFS_BR
    if opcao == "EUA":
        return ETFS_EUA
    return ETFS_BR + ETFS_EUA

# Função para obter preços históricos
def get_prices(tickers, years=5):
    # Usar cache do session_state se disponível
    cache_key = f"prices_cache_{','.join(tickers)}_{years}"
    
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    
    all_data = pd.DataFrame()
    
    with st.spinner(f"Obtendo dados históricos para {len(tickers)} ETFs..."):
        for ticker in tickers:
            try:
                url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={FMP_API_KEY}"
                response = requests.get(url)
                data = response.json()
                
                if "historical" in data:
                    df = pd.DataFrame(data["historical"])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date')
                    
                    # Filtrar para os últimos X anos
                    min_date = pd.Timestamp.now() - pd.DateOffset(years=years)
                    df = df[df['date'] >= min_date]
                    
                    # Manter apenas as colunas necessárias
                    df = df[['date', 'close']]
                    df.columns = ['date', ticker]
                    
                    if all_data.empty:
                        all_data = df
                    else:
                        all_data = pd.merge(all_data, df, on='date', how='outer')
            except Exception as e:
                st.warning(f"Erro ao obter dados para {ticker}: {str(e)}")
    
    # Ordenar por data e definir data como índice
    if not all_data.empty:
        all_data = all_data.sort_values('date')
        all_data.set_index('date', inplace=True)
        
        # Preencher valores faltantes
        all_data = all_data.ffill().bfill()
        
        # Armazenar no cache
        st.session_state[cache_key] = all_data
    
    return all_data

# Função para estimar retornos e covariância
def estimar_retornos_cov(prices_df):
    # Calcular retornos diários
    returns = prices_df.pct_change().dropna()
    
    # Calcular média anualizada (252 dias úteis)
    mu = returns.mean() * 252
    
    # Calcular matriz de covariância anualizada
    S = returns.cov() * 252
    
    return mu, S

# Função para otimizar a carteira
def otimizar(tickers, perfil, prices_df=None):
    from pypfopt import EfficientFrontier, expected_returns, risk_models
    
    if prices_df is None or prices_df.empty:
        prices_df = get_prices(tickers)
    
    if prices_df.empty:
        return {}, (0, 0, 0)
    
    # Calcular retornos esperados e covariância
    mu = expected_returns.mean_historical_return(prices_df)
    S = risk_models.sample_cov(prices_df)
    
    # Criar o objeto EfficientFrontier
    ef = EfficientFrontier(mu, S)
    
    # Adicionar restrições
    ef.add_constraint(lambda w: w >= 0.04)  # Mínimo 4% por ativo
    ef.add_constraint(lambda w: w <= 0.20)  # Máximo 20% por ativo
    
    # Otimizar de acordo com o perfil
    if perfil == "Conservador":
        ef.min_volatility()
    elif perfil == "Moderado":
        ef.max_sharpe()
    else:  # Agressivo
        ef.efficient_risk(target_risk=0.20)
    
    # Limpar os pesos e selecionar os top 10
    pesos = ef.clean_weights()
    top10 = dict(sorted(pesos.items(), key=lambda x: x[1], reverse=True)[:10])
    
    # Normalizar para soma = 1
    total = sum(top10.values())
    top10 = {k: v/total for k, v in top10.items()}
    
    # Calcular performance
    performance = ef.portfolio_performance(verbose=False)
    
    return top10, performance

# Função para gerar análise de texto com OpenAI
def gerar_texto(carteira, perfil):
    prompt = f"""
    Você é analista. Explique esta carteira de 10 ETFs {carteira}
    para um investidor {perfil}, em até 400 palavras, em português simples.
    """
    
    try:
        # Verificar se a API key está disponível
        if not OPENAI_API_KEY:
            return "Não foi possível gerar a análise: Chave de API do OpenAI não configurada."
        
        # Atualizar para usar a nova API da OpenAI
        try:
            from openai import OpenAI
            
            # Remover possíveis prefixos que causam erros
            clean_api_key = OPENAI_API_KEY
            if clean_api_key.startswith("sk-proj-"):
                # Remover prefixo que pode causar o erro "Incorrect API key provided"
                clean_api_key = clean_api_key.replace("sk-proj-", "sk-")
            
            # Criar cliente com a API key limpa
            client = OpenAI(api_key=clean_api_key)
            
            # Fazer a chamada para a API com o novo formato
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Você é um analista financeiro especializado em ETFs e investimentos."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=800
            )
            
            # Extrair e retornar o texto da resposta
            return response.choices[0].message.content.strip()
        except Exception as api_error:
            # Fallback para a API antiga
            try:
                resp = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                return f"Não foi possível gerar a análise: {str(e)}. Você pode encontrar sua API key em https://platform.openai.com/account/api-keys."
    except Exception as e:
        return f"Não foi possível gerar a análise: {str(e)}"

# Função para gerar relatório PDF
def gerar_pdf(df_aloc, analise_texto, perfil, universo, performance):
    from weasyprint import HTML
    import tempfile
    
    # Criar conteúdo HTML para o relatório
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Relatório ETF Blueprint</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 30px; }}
            h1, h2 {{ color: #2C3E50; }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .date {{ font-size: 14px; color: #7F8C8D; margin-bottom: 20px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #BDC3C7; padding: 12px; text-align: left; }}
            th {{ background-color: #F5F7FA; }}
            .performance {{ margin: 20px 0; padding: 15px; background-color: #F8F9F9; border-radius: 5px; }}
            .analysis {{ margin: 20px 0; text-align: justify; line-height: 1.6; }}
            .footer {{ margin-top: 50px; font-size: 12px; color: #95A5A6; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>ETF Blueprint - Relatório de Carteira</h1>
            <div class="date">Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}</div>
        </div>
        
        <h2>Perfil do Investidor</h2>
        <p>Perfil: <strong>{perfil}</strong></p>
        <p>Universo de ETFs: <strong>{universo}</strong></p>
        
        <h2>Carteira Otimizada</h2>
        <table>
            <tr>
                <th>ETF</th>
                <th>Alocação (%)</th>
            </tr>
    """
    
    # Adicionar linhas da tabela
    for index, row in df_aloc.iterrows():
        html_content += f"""
            <tr>
                <td>{row['ETF']}</td>
                <td>{row['Alocação (%)']:.2f}%</td>
            </tr>
        """
    
    # Adicionar resultados de performance
    expected_return, volatility, sharpe = performance
    html_content += f"""
        </table>
        
        <div class="performance">
            <h2>Performance Esperada</h2>
            <p>Retorno Anual Esperado: <strong>{expected_return*100:.2f}%</strong></p>
            <p>Volatilidade Anual: <strong>{volatility*100:.2f}%</strong></p>
            <p>Índice Sharpe: <strong>{sharpe:.2f}</strong></p>
        </div>
        
        <div class="analysis">
            <h2>Análise da Carteira</h2>
            <p>{analise_texto}</p>
        </div>
        
        <div class="footer">
            <p>© {datetime.now().year} ETF Blueprint - Todos os direitos reservados</p>
        </div>
    </body>
    </html>
    """
    
    # Usar temporário para criar o PDF
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
        f.write(html_content.encode('utf-8'))
        temp_path = f.name
    
    # Converter HTML para PDF
    pdf_bytes = HTML(filename=temp_path).write_pdf()
    
    # Remover arquivo temporário
    os.unlink(temp_path)
    
    return pdf_bytes

# Inicializar session_state
if 'stage' not in st.session_state:
    st.session_state['stage'] = "input"

# Título principal
st.title("ETF Blueprint 📈")
st.markdown("Construção de carteiras otimizadas de ETFs")

# Lógica principal do aplicativo
if st.session_state['stage'] == "input":
    # Formulário para perfil de investidor
    with st.form("perfil_form"):
        st.subheader("Perfil do Investidor")
        
        col1, col2 = st.columns(2)
        
        with col1:
            perfil_risco = st.selectbox(
                "Perfil de Risco",
                ["Conservador", "Moderado", "Agressivo"]
            )
            
            horizonte = st.number_input(
                "Horizonte de Investimento (anos)",
                min_value=1,
                max_value=30,
                value=5,
                step=1
            )
        
        with col2:
            max_drawdown = st.slider(
                "Drawdown Máximo Aceitável (%)",
                min_value=5,
                max_value=40,
                value=15,
                step=5
            )
            
            universo = st.radio(
                "Universo de ETFs",
                ["BR", "EUA", "Ambos"]
            )
        
        submit_button = st.form_submit_button("Gerar Carteira")
        
        if submit_button:
            # Salvar perfil no session_state
            st.session_state['perfil'] = {
                'risco': perfil_risco,
                'horizonte': horizonte,
                'max_drawdown': max_drawdown,
                'universo': universo
            }
            
            # Obter lista de ETFs conforme o universo selecionado
            tickers = filtrar_universo(universo)
            
            # Obter preços históricos
            with st.spinner("Obtendo dados históricos e otimizando carteira..."):
                prices_df = get_prices(tickers, horizonte)
                
                if not prices_df.empty:
                    # Otimizar carteira
                    carteira, performance = otimizar(tickers, perfil_risco, prices_df)
                    
                    if carteira:
                        # Criar DataFrame com a alocação
                        df_aloc = pd.DataFrame({
                            'ETF': list(carteira.keys()),
                            'Alocação (%)': [v * 100 for v in carteira.values()]
                        })
                        
                        # Gerar análise de texto
                        analise = gerar_texto(carteira, perfil_risco)
                        
                        # Salvar resultados
                        st.session_state['portfolio_result'] = {
                            'alocacao': df_aloc,
                            'performance': performance,
                            'analise': analise,
                            'carteira': carteira
                        }
                        
                        # Mudar para a tela de resultados
                        st.session_state['stage'] = "result"
                        st.experimental_rerun()
                    else:
                        st.error("Não foi possível otimizar a carteira. Tente outro universo de ETFs.")
                else:
                    st.error("Não foi possível obter dados históricos. Verifique sua conexão e a chave API.")

elif st.session_state['stage'] == "result":
    # Exibir resultados da carteira otimizada
    
    # Recuperar dados
    df_aloc = st.session_state['portfolio_result']['alocacao']
    performance = st.session_state['portfolio_result']['performance']
    analise = st.session_state['portfolio_result']['analise']
    carteira = st.session_state['portfolio_result']['carteira']
    perfil = st.session_state['perfil']
    
    # Mostrar mensagem de sucesso
    st.success("Carteira otimizada com sucesso!")
    
    # Layout em colunas
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Tabela de alocação
        st.subheader("Alocação Recomendada")
        st.dataframe(df_aloc, height=300)
        
        # Métricas de performance
        expected_return, volatility, sharpe = performance
        
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            st.metric("Retorno Esperado", f"{expected_return*100:.2f}%")
        
        with metrics_col2:
            st.metric("Volatilidade", f"{volatility*100:.2f}%")
        
        with metrics_col3:
            st.metric("Índice Sharpe", f"{sharpe:.2f}")
    
    with col2:
        # Gráfico de pizza
        st.subheader("Distribuição da Carteira")
        
        # Garantir que estamos usando o nome da coluna correto
        fig = px.pie(
            df_aloc,
            values='Alocação (%)',  # Confirmar que este é o nome exato da coluna
            names='ETF',
            title='Composição da Carteira'
        )
        
        # Melhorar a formatação dos textos e labels
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            texttemplate='%{label}<br>%{percent:.1%}',
            hovertemplate='<b>%{label}</b><br>Alocação: %{percent:.2%}'
        )
        
        # Atualizar layout para melhor visualização
        fig.update_layout(
            uniformtext_minsize=12,
            uniformtext_mode='hide',
            legend=dict(font=dict(size=10))
        )
        
        st.plotly_chart(fig)
    
    # Análise textual
    st.subheader("Análise da Carteira")
    st.markdown(f"**Perfil: {perfil['risco']} | Universo: {perfil['universo']} | Horizonte: {perfil['horizonte']} anos**")
    st.markdown(analise)
    
    # Botões para download
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        # Download CSV
        csv = df_aloc.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Baixar CSV",
            csv,
            "carteira_etf.csv",
            "text/csv"
        )
    
    with col2:
        # Download PDF
        pdf_bytes = gerar_pdf(
            df_aloc,
            analise,
            perfil['risco'],
            perfil['universo'],
            performance
        )
        
        st.download_button(
            "Baixar PDF",
            pdf_bytes,
            "relatorio_etf_blueprint.pdf",
            "application/pdf"
        )
    
    with col3:
        # Botão para voltar
        if st.button("← Voltar para Formulário"):
            st.session_state['stage'] = "input"
            st.experimental_rerun() 