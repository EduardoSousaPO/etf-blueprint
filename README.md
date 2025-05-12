# ETF Blueprint 📈

Uma aplicação para otimização de carteiras de ETFs usando análise quantitativa e inteligência artificial.

## Sobre o Projeto

O ETF Blueprint é uma ferramenta web que permite aos investidores construir carteiras otimizadas com ETFs do Brasil e dos EUA. O aplicativo utiliza algoritmos de otimização convexa (CVXPY) para criar carteiras eficientes com base no perfil de risco do investidor.

### Principais Funcionalidades

- Otimização de carteira com base em 3 perfis: Conservador, Moderado e Agressivo
- Universo de ETFs do Brasil, EUA ou ambos
- Visualização da alocação recomendada com gráficos interativos
- Análise textual da carteira gerada por IA
- Exportação de resultados em CSV e PDF

## Requisitos

- Python 3.9 (recomendado)
- Bibliotecas Python listadas em `requirements.txt`
- Chaves API:
  - Financial Modeling Prep (FMP) API
  - OpenAI API (opcional, para análise por IA)

## Instalação e Execução

### Windows com WSL (Recomendado)

1. Certifique-se de ter o WSL instalado no Windows
2. Clone este repositório
3. Execute o arquivo `run_app_wsl.bat` com duplo clique
   - Este script automaticamente criará um ambiente virtual Python 3.9
   - Instalará todas as dependências necessárias
   - Iniciará a aplicação Streamlit

### Instalação Manual

1. Clone este repositório
2. Crie um ambiente virtual com Python 3.9:
   ```
   python3.9 -m venv venv39
   source venv39/bin/activate  # Linux/macOS
   venv39\Scripts\activate     # Windows
   ```
3. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```
4. Configure as chaves API:
   - Crie um arquivo `.env` com base no modelo `.env-modelo`
   - Ou utilize o arquivo `.streamlit/secrets.toml`

5. Execute a aplicação:
   ```
   streamlit run streamlit_app.py
   ```

## Deploy

A aplicação está configurada para deploy no Streamlit Cloud. Os arquivos `runtime.txt` e `packages.txt` contêm as configurações necessárias para o ambiente de produção.

## Estrutura do Projeto

- `streamlit_app.py`: Arquivo principal da aplicação
- `.streamlit/`: Configurações do Streamlit
- `backend/`: Serviços e lógica de negócio
- `utils/`: Funções utilitárias
- `assets/`: Arquivos estáticos (imagens, etc.)
- `tests/`: Testes automatizados

## Solução de Problemas

### Erro com CVXPY
Se encontrar problemas com a instalação do CVXPY, a aplicação usará automaticamente uma implementação alternativa de otimização.

### Python 3.12
Evite usar Python 3.12, pois algumas dependências (especialmente CVXPY) podem não ser compatíveis.

## Licença

Este projeto é distribuído sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## Contato

Eduardo Sousa - [@EduardoSousaPO](https://github.com/EduardoSousaPO) 