#Classificação, Clusterização e Sumarização de Relatos Criminais

Este projeto realiza um pipeline completo de análise de **relatos de crimes**, com foco em **roubos de celular**, utilizando técnicas de **classificação supervisionada**, **clusterização não supervisionada** e **sumarização automática com LLMs**.

---

## Visão Geral do Pipeline

O fluxo principal executado no arquivo `main_pipeline.py` é dividido em três etapas:

### 1. Classificação Supervisionada (`classification.py`)
- Classifica os relatos de crime com base no campo `especificacao_crime`.
- Gera rótulos como:
  - `0` — Transeunte
  - `1` — Celular
  - `2` — Moto
  - `3` — Em coletivo
  - `4` — Outros
- Utiliza TF-IDF + `LinearSVC` para treinamento.
- Avalia desempenho com métricas e exibe matrizes de confusão e gráficos de linha.

### 2. Clusterização de Relatos de Celular (`clustering.py`)
- Filtra relatos classificados como "celular".
- Aplica TF-IDF + KMeans.
- Determina o número ideal de clusters com o **método do cotovelo**.
- Gera visualização 2D com PCA.
- Exporta um CSV com `relato_limpo` e `cluster`.

### 3. Sumarização por Cluster com LLM (`summarization.py`)
- Utiliza `LangChain`, `Ollama` e o modelo local `llama3.1:latest`.
- Embeddings gerados com `BAAI/bge-base-en-v1.5`.
- Fragmenta relatos por cluster, aplica clustering semântico e gera resumos com prompt customizado.
- Retorna um resumo explicativo para cada cluster detectado.

---

## Estrutura dos Arquivos

```plaintext
.
├── base_roubo.csv             # Base de entrada com os relatos
├── main_pipeline.py           # Executa o pipeline completo
├── classification.py          # Etapa de classificação com SVM
├── clustering.py              # Clusterização dos relatos de celular
├── summarization.py           # Sumarização usando LLM local (LangChain + Ollama)
├── cluster_X.csv              # Gerado após clusterização (usado na sumarização)
├── requirements.txt           # Dependências do projeto
```

---

## Como Executar

### 1. Clonar o repositório

### Bash
git clone https://github.com/seu-usuario/seu-repo.git
cd seu-repo

### Criar ambiente virtual (opcional)
python -m venv .venv
source .venv/bin/activate   # No Linux/macOS
.venv\Scripts\activate 

### Instalar as dependências 
pip install -r requirements.txt

### Rodar o pipeline completo
python main_pipeline.py

