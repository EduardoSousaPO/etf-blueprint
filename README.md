# ETF Blueprint 📈

Uma aplicação para otimização de carteiras de ETFs personalizadas com base no perfil do investidor.

## Sobre o Projeto

ETF Blueprint é uma ferramenta que permite aos investidores criar carteiras de ETFs (Exchange Traded Funds) globais otimizadas em questão de minutos. A aplicação analisa dados históricos, perfil de risco e objetivos do investidor para recomendar a melhor alocação possível.

## Funcionalidades

- 📊 Otimização de carteira baseada em retorno esperado, volatilidade e sharpe ratio
- 🎯 Criação de carteira otimizada de exatos 10 ETFs com alocação entre 4% e 20% por ativo
- 🧠 Análise personalizada com narrativa gerada por IA (OpenAI API)
- 📈 Visualização com gráficos Plotly interativos
- 📄 Geração de relatório PDF detalhado
- 💼 Exportação dos dados em CSV para implementação em corretoras

## Tecnologias

- Python 3.9+
- Streamlit (frontend)
- Financial Modeling Prep API (dados financeiros)
- PyPortfolioOpt (otimização de portfólios)
- OpenAI API (geração de narrativas personalizadas)
- Plotly (visualizações)
- WeasyPrint (geração de PDF)

## Requisitos

- Python 3.9+
- API Key da Financial Modeling Prep (FMP)
- API Key da OpenAI

## Configuração

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/etf-blueprint.git
cd etf-blueprint
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Configure as variáveis de ambiente:

Crie um arquivo `.env` na raiz do projeto com as seguintes variáveis:
```
FMP_API_KEY=sua_chave_fmp_api
OPENAI_API_KEY=sua_chave_openai_api
```

Alternativamente, no Streamlit Cloud, adicione essas chaves em Secrets.

## Execução Local

```bash
streamlit run streamlit_app.py
```

## Deploy no Streamlit Cloud

1. Faça fork deste repositório para sua conta GitHub
2. Acesse [streamlit.io/cloud](https://streamlit.io/cloud) e faça login
3. Clique em "New app" e selecione o repositório
4. Configure as seguintes opções:
   - Main file path: `streamlit_app.py`
   - Python version: 3.9+
5. Configure os Secrets com suas chaves de API:
```toml
FMP_API_KEY = "sua_chave_fmp_api"
OPENAI_API_KEY = "sua_chave_openai_api"
```
6. Clique em "Deploy!"

## Estrutura do Projeto

- `streamlit_app.py`: Aplicativo principal em página única
- `backend/services/`: Serviços para obtenção de dados, otimização e análise
  - `fmp_service.py`: Cliente para API da Financial Modeling Prep
  - `optimizer.py`: Lógica de otimização de carteiras
  - `openai_service.py`: Cliente para API da OpenAI
- `utils/`: Utilitários
  - `pdf_report.py`: Geração de relatórios em PDF
- `tests/`: Testes automatizados
- `requirements.txt`: Dependências do projeto
- `.streamlit/`: Configurações do Streamlit

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes.

## Contato

Eduardo Sousa - [@EduardoSousaPO](https://github.com/EduardoSousaPO) 