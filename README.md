# ETF Blueprint 📈

Uma aplicação para otimização de carteiras de ETFs personalizadas com base no perfil do investidor.

## Sobre o Projeto

ETF Blueprint é uma ferramenta que permite aos investidores criar carteiras de ETFs (Exchange Traded Funds) globais otimizadas em questão de minutos. A aplicação analisa dados históricos, perfil de risco e objetivos do investidor para recomendar a melhor alocação possível.

## Funcionalidades

- 📊 Otimização de carteira baseada em retorno esperado, volatilidade e drawdown
- 🧠 Análise personalizada com narrativa gerada por IA
- 📈 Visualização da fronteira eficiente e comparação com diferentes alocações
- 📄 Geração de relatório PDF detalhado
- 💼 Exportação dos dados em CSV para implementação em corretoras

## Tecnologias

- Python 3.8+
- Streamlit (frontend)
- Financial Modeling Prep API (dados financeiros)
- Biblioteca Scipy/Numpy (otimização de portfólios)
- OpenAI API (geração de narrativas personalizadas)
- Matplotlib/Plotly (visualizações)
- ReportLab (geração de PDF)

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/EduardoSousaPO/etf-blueprint.git
cd etf-blueprint
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Configure as variáveis de ambiente (ou crie um arquivo .env):
```
FMP_API_KEY=sua_chave_api_fmp
OPENAI_API_KEY=sua_chave_api_openai
```

4. Execute a aplicação:
```bash
streamlit run app.py
```

## Uso

1. Acesse a interface web (http://localhost:8501 por padrão)
2. Preencha seu perfil de risco e objetivos financeiros
3. Receba sua carteira otimizada com análise detalhada
4. Baixe o relatório ou exporte os dados para implementação

## Licença

MIT License

## Contato

Eduardo Sousa - [@EduardoSousaPO](https://github.com/EduardoSousaPO) 