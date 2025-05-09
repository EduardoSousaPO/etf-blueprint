import os
import tempfile
from datetime import datetime
import pandas as pd
import plotly.express as px
from weasyprint import HTML
import streamlit as st
import base64

def gerar_pdf(df_aloc, analise_texto, perfil, universo, performance):
    """
    Gera um relatório PDF com a análise da carteira de ETFs.
    
    Args:
        df_aloc (pd.DataFrame): DataFrame com alocações de ETFs.
        analise_texto (str): Texto da análise gerada pela OpenAI.
        perfil (str): Perfil do investidor.
        universo (str): Universo de ETFs escolhido.
        performance (tuple): Tupla com (retorno, volatilidade, sharpe).
        
    Returns:
        bytes: Conteúdo do PDF em bytes para download.
    """
    # Extrair métricas de performance
    expected_return, volatility, sharpe = performance
    
    # Criar conteúdo HTML para o relatório
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Relatório ETF Blueprint</title>
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                margin: 30px; 
                color: #333;
                line-height: 1.6;
            }}
            h1, h2, h3 {{ 
                color: #2C3E50; 
                margin-top: 20px;
            }}
            .header {{ 
                text-align: center; 
                margin-bottom: 30px; 
                border-bottom: 1px solid #eee;
                padding-bottom: 20px;
            }}
            .date {{ 
                font-size: 14px; 
                color: #7F8C8D; 
                margin-bottom: 20px; 
            }}
            table {{ 
                width: 100%; 
                border-collapse: collapse; 
                margin: 20px 0; 
            }}
            th, td {{ 
                border: 1px solid #BDC3C7; 
                padding: 12px; 
                text-align: left; 
            }}
            th {{ 
                background-color: #F5F7FA; 
                font-weight: bold;
            }}
            .performance {{ 
                margin: 20px 0; 
                padding: 15px; 
                background-color: #F8F9F9; 
                border-radius: 5px; 
                border-left: 5px solid #3498DB;
            }}
            .analysis {{ 
                margin: 20px 0; 
                text-align: justify; 
                line-height: 1.6; 
                background-color: #FAFAFA;
                padding: 15px;
                border-radius: 5px;
            }}
            .footer {{ 
                margin-top: 50px; 
                padding-top: 20px;
                font-size: 12px; 
                color: #95A5A6; 
                text-align: center; 
                border-top: 1px solid #eee;
            }}
            .metrics {{ 
                display: flex; 
                justify-content: space-between; 
                margin: 20px 0; 
            }}
            .metric {{ 
                flex: 1; 
                text-align: center; 
                padding: 15px; 
                background-color: #F5F7FA; 
                margin: 0 10px; 
                border-radius: 5px; 
                box-shadow: 0 1px 3px rgba(0,0,0,0.1); 
            }}
            .metric-value {{ 
                font-size: 24px; 
                font-weight: bold; 
                color: #2980B9; 
                margin: 10px 0; 
            }}
            .metric-label {{ 
                font-size: 14px; 
                color: #7F8C8D; 
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>ETF Blueprint - Relatório de Carteira</h1>
            <div class="date">Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}</div>
        </div>
        
        <h2>Perfil do Investidor</h2>
        <p>Perfil de Risco: <strong>{perfil}</strong></p>
        <p>Universo de ETFs: <strong>{universo}</strong></p>
        
        <div class="performance">
            <h2>Performance Esperada</h2>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-label">Retorno Anual Esperado</div>
                    <div class="metric-value">{expected_return*100:.2f}%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Volatilidade Anual</div>
                    <div class="metric-value">{volatility*100:.2f}%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Índice Sharpe</div>
                    <div class="metric-value">{sharpe:.2f}</div>
                </div>
            </div>
        </div>
        
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
    
    # Continuar o HTML
    html_content += f"""
        </table>
        
        <div class="analysis">
            <h2>Análise da Carteira</h2>
            <p>{analise_texto}</p>
        </div>
        
        <div class="footer">
            <p>© {datetime.now().year} ETF Blueprint - Todos os direitos reservados</p>
            <p>Este relatório foi gerado automaticamente e não constitui recomendação de investimento.</p>
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

def get_pie_chart_base64(df_aloc):
    """
    Gera um gráfico de pizza da alocação e retorna como imagem base64.
    
    Args:
        df_aloc (pd.DataFrame): DataFrame com alocações de ETFs.
        
    Returns:
        str: String base64 da imagem do gráfico.
    """
    fig = px.pie(
        df_aloc, 
        values='Alocação (%)', 
        names='ETF',
        title='Composição da Carteira',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        margin=dict(t=50, b=0, l=0, r=0),
        height=400,
        width=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    # Salvar em arquivo temporário
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        fig.write_image(f.name, format='png')
        temp_path = f.name
    
    # Converter para base64
    with open(temp_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    
    # Remover arquivo temporário
    os.unlink(temp_path)
    
    return encoded_string 