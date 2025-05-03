import os
import io
import base64
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Para funcionar em ambientes sem GUI
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    Image, PageBreak, ListFlowable, ListItem
)
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.legends import Legend
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from io import BytesIO

class PDFGenerator:
    """
    Serviço para geração de relatórios em PDF para carteiras de ETFs
    """
    
    def __init__(self, logo_path: Optional[str] = None):
        """
        Inicializa o gerador de PDF
        
        Args:
            logo_path: Caminho para o logo a ser exibido no cabeçalho do PDF (opcional)
        """
        # Verificar dependências necessárias
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            from reportlab.lib.pagesizes import A4
        except ImportError as e:
            error_msg = str(e)
            if "matplotlib" in error_msg:
                raise ImportError("Biblioteca matplotlib não encontrada. Execute 'pip install matplotlib' para instalá-la.")
            elif "reportlab" in error_msg:
                raise ImportError("Biblioteca reportlab não encontrada. Execute 'pip install reportlab' para instalá-la.")
            else:
                raise ImportError(f"Dependência não encontrada: {error_msg}")
            
        self.logo_path = logo_path or os.path.join("assets", "logo.png")
        
        # Verificar se o arquivo do logo existe
        if not os.path.exists(self.logo_path):
            # Tentar caminhos alternativos
            alt_paths = [
                os.path.join(".", "assets", "logo.png"),
                os.path.join("..", "assets", "logo.png"),
                os.path.join("assets", "logo_small.png"),
                os.path.join(".", "assets", "logo_small.png")
            ]
            
            for path in alt_paths:
                if os.path.exists(path):
                    self.logo_path = path
                    break
        
        # Definir estilos
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Configura estilos personalizados para o PDF"""
        # Estilo para título
        self.styles.add(ParagraphStyle(
            name='Title',
            parent=self.styles['Title'],
            fontSize=24,
            leading=28,
            textColor=colors.HexColor('#1E3A8A'),
            spaceAfter=20
        ))
        
        # Estilo para subtítulo
        self.styles.add(ParagraphStyle(
            name='Heading2',
            parent=self.styles['Heading2'],
            fontSize=16,
            leading=20,
            textColor=colors.HexColor('#1E3A8A'),
            spaceBefore=15,
            spaceAfter=10
        ))
        
        # Estilo para texto normal
        self.styles.add(ParagraphStyle(
            name='Body',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=14,
            alignment=TA_JUSTIFY
        ))
        
        # Estilo para descrição da carteira
        self.styles.add(ParagraphStyle(
            name='Narrative',
            parent=self.styles['Normal'],
            fontSize=12,
            leading=16,
            spaceBefore=10,
            spaceAfter=10,
            backColor=colors.HexColor('#F5F5F5'),
            borderPadding=10,
            alignment=TA_JUSTIFY
        ))
        
        # Estilo para rodapé
        self.styles.add(ParagraphStyle(
            name='Footer',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.gray
        ))
        
        # Estilo para cabeçalho de tabela
        self.styles.add(ParagraphStyle(
            name='TableHeader',
            parent=self.styles['Normal'],
            fontSize=10,
            alignment=TA_CENTER,
            textColor=colors.white
        ))
    
    def generate(
        self, 
        perfil: Dict[str, Any], 
        carteira: Any,  # Aceitar qualquer tipo
        metricas: Dict[str, float], 
        narrativa: str
    ) -> bytes:
        """
        Gera o relatório PDF da carteira de ETFs
        
        Args:
            perfil: Dados do perfil do investidor
            carteira: Dados da carteira otimizada (dict ou objeto)
            metricas: Métricas da carteira
            narrativa: Texto explicativo da carteira
            
        Returns:
            Bytes do PDF gerado
        """
        # Converter carteira para dicionário se for um objeto
        if hasattr(carteira, '__dict__'):
            carteira_dict = carteira.__dict__.copy()
            # Se tiver um campo 'etfs', garantir que seja uma lista de dicts
            if 'etfs' in carteira_dict and isinstance(carteira_dict['etfs'], list):
                etfs_list = []
                for etf in carteira_dict['etfs']:
                    if hasattr(etf, '__dict__'):
                        etfs_list.append(etf.__dict__)
                    elif isinstance(etf, dict):
                        etfs_list.append(etf)
                carteira_dict['etfs'] = etfs_list
        else:
            carteira_dict = carteira
            
        # Buffer para armazenar o PDF
        buffer = io.BytesIO()
        
        # Criar documento
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )
        
        # Lista de elementos do PDF
        elements = []
        
        # Adicionar cabeçalho
        self._add_header(elements, perfil)
        
        # Adicionar resumo do perfil
        self._add_perfil_summary(elements, perfil)
        
        # Adicionar métricas principais
        self._add_metricas(elements, metricas)
        
        # Adicionar narrativa
        self._add_narrativa(elements, narrativa)
        
        # Adicionar tabela de alocação
        self._add_tabela_alocacao(elements, carteira_dict)
        
        # Adicionar gráficos
        self._add_graficos(elements, carteira_dict)
        
        # Adicionar recomendações de implementação
        self._add_recomendacoes(elements)
        
        # Adicionar rodapé
        self._add_footer(elements)
        
        # Construir o documento
        doc.build(elements, onFirstPage=self._add_page_number, onLaterPages=self._add_page_number)
        
        # Retornar bytes
        buffer.seek(0)
        return buffer.getvalue()
    
    def _add_header(self, elements: List, perfil: Dict[str, Any]):
        """Adiciona cabeçalho ao PDF"""
        # Título
        elements.append(Paragraph("Carteira Personalizada de ETFs", self.styles['Title']))
        
        # Nome do cliente e data
        nome_cliente = perfil.get('nome', 'Cliente')
        data_atual = datetime.now().strftime("%d/%m/%Y")
        elements.append(
            Paragraph(
                f"<para alignment='right'><b>Cliente:</b> {nome_cliente}<br/>"
                f"<b>Data:</b> {data_atual}</para>", 
                self.styles['Normal']
            )
        )
        
        elements.append(Spacer(1, 10))
        
        # Linha separadora
        elements.append(
            Paragraph("<hr color='#1E3A8A' thickness='2'/>", self.styles['Normal'])
        )
        
        elements.append(Spacer(1, 10))
    
    def _add_perfil_summary(self, elements: List, perfil: Dict[str, Any]):
        """Adiciona resumo do perfil do investidor"""
        elements.append(Paragraph("Seu Perfil de Investidor", self.styles['Heading2']))
        
        # Criar tabela com dados do perfil
        data = [
            ["Horizonte de Investimento", f"{perfil.get('horizonte', 'N/A')} anos"],
            ["Perfil de Risco", perfil.get('perfil_risco', 'N/A')],
            ["Tolerância a Drawdown", perfil.get('tolerancia_drawdown', 'N/A')],
            ["Retorno Alvo", f"{perfil.get('retorno_alvo', 'N/A')}%"],
            ["Objetivos", ", ".join(perfil.get('objetivos', ['N/A']))],
            ["Regiões Preferidas", ", ".join(perfil.get('regioes', ['Global']))]
        ]
        
        # Estilo da tabela de perfil
        table = Table(data, colWidths=[4*cm, 11*cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E5E7EB')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1E3A8A')),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 15))
    
    def _add_metricas(self, elements: List, metricas: Dict[str, float]):
        """Adiciona métricas principais da carteira"""
        elements.append(Paragraph("Métricas da Carteira", self.styles['Heading2']))
        
        # Criar tabela de métricas
        data = [
            ["Métrica", "Valor"],
            ["Retorno Esperado", f"{metricas.get('retorno_esperado', 'N/A')}%"],
            ["Volatilidade", f"{metricas.get('volatilidade', 'N/A')}%"],
            ["Índice Sharpe", f"{metricas.get('sharpe_ratio', 'N/A')}"],
            ["Drawdown Máximo", f"{metricas.get('max_drawdown', 'N/A')}%"]
        ]
        
        # Estilo da tabela de métricas
        table = Table(data, colWidths=[7.5*cm, 7.5*cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#F5F5F5')),
            ('BACKGROUND', (1, 1), (1, -1), colors.white),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('ALIGN', (1, 1), (1, -1), 'CENTER'),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 1), (1, -1), 'Helvetica'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 15))
    
    def _add_narrativa(self, elements: List, narrativa: str):
        """Adiciona texto narrativo sobre a carteira"""
        elements.append(Paragraph("Análise da Carteira", self.styles['Heading2']))
        
        # Remover espaços extras e quebras de linha
        narrativa = ' '.join([line.strip() for line in narrativa.split('\n') if line.strip()])
        
        elements.append(Paragraph(narrativa, self.styles['Narrative']))
        elements.append(Spacer(1, 15))
    
    def _add_tabela_alocacao(self, elements: List, carteira: Dict[str, Any]):
        """Adiciona tabela de alocação da carteira"""
        elements.append(Paragraph("Composição da Carteira", self.styles['Heading2']))
        
        # Verificar a estrutura da carteira para obter os ETFs
        etfs = []
        if isinstance(carteira, dict):
            if 'etfs' in carteira and isinstance(carteira['etfs'], list):
                etfs = carteira['etfs']
            elif 'portfolio' in carteira and isinstance(carteira['portfolio'], list):
                etfs = carteira['portfolio']
        elif isinstance(carteira, list):
            etfs = carteira
        
        # Se não houver ETFs, mostrar mensagem
        if not etfs:
            elements.append(Paragraph("Não foram encontrados dados de ETFs na carteira.", self.styles['Body']))
            return
        
        # Preparar dados para a tabela
        headers = ["Ticker", "Nome", "Categoria", "Alocação", "Tx. Adm."]
        data = [headers]
        
        # Preencher dados
        for etf in etfs:
            if isinstance(etf, dict):
                ticker = etf.get('symbol', 'N/A')
                nome = etf.get('name', 'N/A')
                categoria = etf.get('categoria', 'N/A')
                peso = f"{etf.get('peso', 0):.2f}%"
                expense_ratio = f"{etf.get('expense_ratio', 0):.2f}%" if 'expense_ratio' in etf else 'N/A'
                
                data.append([ticker, nome, categoria, peso, expense_ratio])
        
        # Criar tabela
        colWidths = [2*cm, 6*cm, 3*cm, 2*cm, 2*cm]
        table = Table(data, colWidths=colWidths)
        
        # Estilo da tabela
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('ALIGN', (1, 1), (1, -1), 'LEFT'),
            ('ALIGN', (2, 1), (2, -1), 'LEFT'),
            ('ALIGN', (3, 1), (3, -1), 'RIGHT'),
            ('ALIGN', (4, 1), (4, -1), 'RIGHT'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 15))
    
    def _add_graficos(self, elements: List, carteira: Dict[str, Any]):
        """Adiciona gráficos ao PDF"""
        elements.append(Paragraph("Visualização da Carteira", self.styles['Heading2']))
        
        # Verificar a estrutura da carteira para obter os ETFs
        etfs = []
        if isinstance(carteira, dict):
            if 'etfs' in carteira and isinstance(carteira['etfs'], list):
                etfs = carteira['etfs']
            elif 'portfolio' in carteira and isinstance(carteira['portfolio'], list):
                etfs = carteira['portfolio']
        elif isinstance(carteira, list):
            etfs = carteira
        
        # Se não houver ETFs, mostrar mensagem
        if not etfs:
            elements.append(Paragraph("Não foram encontrados dados para gerar gráficos.", self.styles['Body']))
            return
        
        # Preparar dados para o gráfico de pizza
        labels = []
        sizes = []
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for i, etf in enumerate(etfs):
            if isinstance(etf, dict):
                ticker = etf.get('symbol', f'ETF {i+1}')
                peso = etf.get('peso', 0)
                
                labels.append(ticker)
                sizes.append(peso)
        
        # Criar gráfico de pizza
        plt.figure(figsize=(8, 5))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Distribuição da Carteira por ETF')
        
        # Salvar gráfico em um buffer
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150)
        img_buffer.seek(0)
        plt.close()
        
        # Adicionar imagem ao PDF
        img = Image(img_buffer, width=16*cm, height=10*cm)
        elements.append(img)
        elements.append(Spacer(1, 15))
        
        # Adicionar gráfico de barras por categoria (se disponível)
        try:
            # Agrupar por categoria se disponível
            categorias = {}
            for etf in etfs:
                if isinstance(etf, dict) and 'categoria' in etf and 'peso' in etf:
                    categoria = etf['categoria']
                    peso = etf['peso']
                    
                    if categoria in categorias:
                        categorias[categoria] += peso
                    else:
                        categorias[categoria] = peso
            
            # Se tivermos categorias, criar o gráfico
            if categorias:
                cat_labels = list(categorias.keys())
                cat_sizes = list(categorias.values())
                
                plt.figure(figsize=(8, 5))
                plt.bar(cat_labels, cat_sizes, color=colors[:len(cat_labels)])
                plt.title('Alocação por Categoria de Ativo')
                plt.ylabel('Alocação (%)')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                # Salvar gráfico em um buffer
                cat_img_buffer = BytesIO()
                plt.savefig(cat_img_buffer, format='png', dpi=150)
                cat_img_buffer.seek(0)
                plt.close()
                
                # Adicionar imagem ao PDF
                elements.append(Paragraph("Alocação por Categoria", self.styles['Heading2']))
                cat_img = Image(cat_img_buffer, width=16*cm, height=10*cm)
                elements.append(cat_img)
                elements.append(Spacer(1, 15))
        except Exception as e:
            # Se houver erro ao gerar o segundo gráfico, apenas ignorar
            pass
    
    def _add_recomendacoes(self, elements: List):
        """Adiciona recomendações de implementação"""
        elements.append(Paragraph("Recomendações de Implementação", self.styles['Heading2']))
        
        # Lista de recomendações
        recomendacoes = [
            "Implemente a alocação através de uma corretora que ofereça acesso a ETFs internacionais com custos baixos.",
            "Rebalanceie a carteira anualmente ou quando os pesos desviarem mais de 5% do alvo.",
            "Considere implementar a carteira em etapas, especialmente em períodos de alta volatilidade.",
            "Mantenha uma reserva de emergência separada antes de investir na carteira de ETFs.",
            "Revise seu plano financeiro e objetivos anualmente para garantir que a carteira ainda está alinhada."
        ]
        
        lista = []
        for rec in recomendacoes:
            lista.append(ListItem(Paragraph(rec, self.styles['Body'])))
        
        elements.append(ListFlowable(lista, bulletType='bullet', leftIndent=20))
        elements.append(Spacer(1, 20))
    
    def _add_footer(self, elements: List):
        """Adiciona rodapé ao PDF"""
        footer_text = (
            "Este relatório foi gerado automaticamente pelo ETF Blueprint e serve apenas como sugestão de "
            "investimento. Os retornos históricos não garantem resultados futuros. Recomendamos consultar "
            "um profissional qualificado antes de tomar decisões de investimento."
        )
        
        elements.append(
            Paragraph("<hr color='#1E3A8A' thickness='1'/>", self.styles['Normal'])
        )
        
        elements.append(Spacer(1, 5))
        elements.append(Paragraph(footer_text, self.styles['Footer']))
    
    def _add_page_number(self, canvas, doc):
        """Adiciona número de página e logo ao cabeçalho"""
        try:
            # Salvar estado
            canvas.saveState()
            
            # Adicionar logo
            if os.path.exists(self.logo_path):
                # Posicionar no canto superior direito
                # Ajustar dimensões para não ficar muito grande
                logo_width = 1.5*cm
                logo_height = 1.5*cm
                x = doc.pagesize[0] - 2*cm - logo_width  # A4 width - margin - logo_width
                y = doc.pagesize[1] - 1.5*cm - logo_height  # A4 height - margin - logo_height
                canvas.drawImage(self.logo_path, x, y, width=logo_width, height=logo_height)
            
            # Adicionar número de página no rodapé
            page_num = canvas.getPageNumber()
            text = f"Página {page_num}"
            
            # Centralizar na largura da página
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(colors.grey)
            canvas.drawCentredString(doc.pagesize[0] / 2, 1*cm, text)
            
            # Adicionar timestamp no rodapé esquerdo
            timestamp = datetime.now().strftime("%d/%m/%Y %H:%M")
            canvas.drawString(2*cm, 1*cm, timestamp)
            
            # Restaurar estado
            canvas.restoreState()
        except Exception as e:
            # Em caso de erro, apenas ignorar e não adicionar número de página
            # para não interromper a geração do PDF
            print(f"Erro ao adicionar número de página: {str(e)}")
            canvas.restoreState() 