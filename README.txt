# ⚖️ LexInsight: Sentimento & Dados

> Transformando linguagem jurídica em inteligência estratégica por meio de NLP e análise orientada a dados.

---

## 🚀 Visão Geral

O **LexInsight** é uma aplicação analítica desenvolvida para extrair insights relevantes a partir de textos jurídicos, utilizando técnicas modernas de **Processamento de Linguagem Natural (NLP)**.

A solução automatiza a leitura, interpretação e estruturação de documentos, convertendo conteúdo textual complexo em **informação estratégica utilizável**.

---

## 🎯 Problema

A análise de documentos jurídicos é:

- Demorada  
- Subjetiva  
- Pouco escalável  
- Dependente de leitura manual  

Isso dificulta a identificação de padrões, riscos e oportunidades em larga escala.

---

## 💡 Solução

O LexInsight propõe uma abordagem orientada a dados:

- Estrutura textos não estruturados  
- Aplica NLP com spaCy  
- Realiza análise de sentimento baseada em léxico jurídico  
- Gera visualizações e métricas objetivas  

---

## 🧠 Proposta de Valor

- **📊 Clareza Analítica** → transforma texto em dados interpretáveis  
- **🧠 Inteligência Jurídica** → evidencia padrões semânticos  
- **⚡ Eficiência Operacional** → reduz tempo de análise manual  
- **📈 Suporte à Decisão** → fornece indicadores estratégicos  

---

## 🧩 Funcionalidades

### 🔹 Processamento de Texto
- Limpeza com Regex  
- Tokenização  
- Lematização (spaCy)  
- Remoção de stopwords (incluindo termos jurídicos)

### 🔹 Análise de Sentimento
- Score contínuo (-1 a +1)  
- Classificação: positivo, negativo, neutro  
- Base léxica adaptada ao domínio jurídico  

### 🔹 Visualização de Dados
- ☁️ WordCloud (frequência de termos)  
- 📊 Distribuição de sentimentos  
- 📉 Frequência lexical  

### 🔹 Interface Interativa
- Upload de textos  
- Processamento em tempo real  
- Dashboard via Streamlit  

---

## ⚙️ Arquitetura da Solução

---

## 🛠️ Tecnologias Utilizadas

- Python  
- Streamlit  
- spaCy  
- Pandas  
- Matplotlib  
- WordCloud  

---

## ▶️ Como Executar

```bash
git clone https://github.com/lucianoadm/LexInsight.git
cd LexInsight
pip install -r requirements.txt
streamlit run app.py
