# ETF Blueprint

[![ETF Blueprint CI](https://github.com/seu-usuario/projeto-ETF-PDF/actions/workflows/python-app.yml/badge.svg)](https://github.com/seu-usuario/projeto-ETF-PDF/actions/workflows/python-app.yml)

ETF Blueprint é uma aplicação web que gera carteiras de ETFs personalizadas com base no perfil de risco do investidor, utilizando o algoritmo de otimização de Markowitz. Além disso, a aplicação gera uma narrativa em linguagem clara explicando a carteira e permite exportar os resultados em PDF e CSV.

## 📋 Funcionalidades

- Formulário de perfil de risco personalizado
- Otimização de carteira com algoritmo mean-variance
- Integração com a API Financial Modeling Prep para dados financeiros
- Geração de narrativa explicativa com OpenAI
- Visualização interativa da carteira com gráficos
- Exportação da carteira em PDF e CSV
- Deploy facilitado na Vercel

## 🚀 Instalação e Execução Local

### Pré-requisitos

- Python 3.11 ou superior
- Conta na [Financial Modeling Prep](https://financialmodelingprep.com/)
- Conta na [OpenAI](https://platform.openai.com/)

### Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/projeto-ETF-PDF.git
cd projeto-ETF-PDF
```

2. Crie e ative um ambiente virtual:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python -m venv venv
source venv/bin/activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Configure as variáveis de ambiente:
```bash
# Windows
copy .env.example .env

# Linux/macOS
cp .env.example .env
```

5. Edite o arquivo `.env` e adicione suas chaves de API:
```
FMP_API_KEY=sua_chave_fmp_aqui
OPENAI_API_KEY=sua_chave_openai_aqui
```

### Execução

Execute a aplicação localmente:
```bash
streamlit run app.py
```

A aplicação estará disponível em `http://localhost:8501`.

## 🚢 Deploy na Vercel

O ETF Blueprint pode ser facilmente implantado na Vercel. Siga estes passos:

1. Instale a CLI da Vercel:
```bash
npm i -g vercel
```

2. Faça login e link do projeto:
```bash
vercel login
vercel link
```

3. Configure as variáveis de ambiente:
```bash
vercel env add FMP_API_KEY
vercel env add OPENAI_API_KEY
```

4. Implante o projeto:
```bash
vercel --prod
```

## 🧪 Testes

Execute os testes automatizados:
```bash
pytest
```

Para gerar um relatório de testes:
```bash
pytest --html=src/backend/tests/report.html
```

## 📊 Exemplo de Uso da API

O backend inclui uma API FastAPI que pode ser acessada em `api/` no deploy da Vercel. Exemplos de uso:

```python
import requests
import json

# Obter otimização de carteira
data = {
    "perfil": {
        "horizonte": 10,
        "perfil_risco": "Moderado",
        "retorno_alvo": 8.5
    },
    "simbolos": ["VTI", "BND", "VEA", "VWO", "GLD"]
}

response = requests.post("https://sua-app.vercel.app/api/optimize", json=data)
carteira = response.json()
```

## 📝 Integração com Hotmart

Para integrar com a Hotmart e gerar relatórios automaticamente após a compra:

1. Configure o webhook da Hotmart para apontar para `https://sua-app.vercel.app/api/hotmart-webhook`.
2. A aplicação irá gerar o PDF e enviar um e-mail com o link para download quando receber a notificação de compra aprovada.

## 🏗️ Estrutura do Projeto

```
projeto-ETF-PDF/
├── app.py                   # Ponto de entrada Streamlit
├── requirements.txt         # Dependências
├── vercel.json              # Configuração de deploy
├── .env.example             # Modelo de variáveis de ambiente
├── .github/                 # Configurações GitHub Actions
│   └── workflows/
│       └── python-app.yml   # CI pipeline
├── assets/                  # Imagens e arquivos estáticos
├── src/                     # Código-fonte
│   ├── backend/             # Lógica de negócios
│   │   ├── models/          # Modelos de dados
│   │   ├── services/        # Serviços (FMP, Optimizer, etc.)
│   │   └── tests/           # Testes automatizados
│   └── frontend/            # Interface de usuário
│       └── pages/           # Páginas Streamlit
```

## 📄 Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes.

## 🙏 Agradecimentos

- [Markowitz Portfolio Theory](https://en.wikipedia.org/wiki/Modern_portfolio_theory)
- [Financial Modeling Prep](https://financialmodelingprep.com/)
- [OpenAI](https://openai.com/)
- [Streamlit](https://streamlit.io/)
- [Vercel](https://vercel.com/) 