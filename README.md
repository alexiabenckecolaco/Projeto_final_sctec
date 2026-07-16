# Pipeline Preditivo de Risco de Crédito

> Projeto Final  
> Autora: Alexia Colaco  
> Dataset: Credit Risk Dataset (32.581 registros, 12 variáveis)

---

## Problema de Negócio

Um banco precisa prever automaticamente se um cliente se tornará **inadimplente** (`loan_status = 1`) ou pagará o empréstimo em dia (`loan_status = 0`). A decisão de conceder ou negar crédito tem impacto financeiro direto:

- **Falso Negativo** (prever "bom pagador" para quem vai calotear) → perda do capital emprestado — o pior cenário.
- **Falso Positivo** (negar crédito a quem pagaria) → perda de receita de juros — ruim, mas recuperável.

O objetivo é construir um modelo que minimize especialmente os Falsos Negativos, protegendo o banco de perdas irreversíveis.

---

## Estrutura do Pipeline

| Fase | Descrição |
|------|-----------|
| 1 | Análise Exploratória de Dados (EDA) |
| 2 | Tratamento e Limpeza (Data Prep) |
| 3 | Feature Engineering |
| 4 | Separação, Balanceamento e Escalonamento |
| 5 | Modelagem e Validação |
| 6 | Avaliação e Veredito de Negócios |

---

## Principais Insights da EDA

**Qualidade dos dados:**
- 5 registros com `person_age > 110` anos (provável erro de digitação) → removidos.
- `loan_int_rate` com 3.116 nulos → imputados com a média (desvio padrão baixo, distribuição simétrica).
- `person_emp_length` com 895 nulos → linhas removidas (sem tempo de financiamento, dado sem valor preditivo claro).
- `person_income` com valores até R$6 milhões → limitado a < R$500.000 para remover outliers extremos.

**Correlações relevantes (Pearson):**
- `person_age` ↔ `cb_person_cred_hist_length` = **0.86** — pessoas mais velhas têm histórico de crédito mais longo (esperado).
- `loan_amnt` ↔ `loan_percent_income` = **0.57** — empréstimos maiores comprometem mais a renda.
- `loan_percent_income` ↔ `loan_status` = **0.38** — maior comprometimento de renda está associado à inadimplência.
- `person_income` ↔ `loan_status` = **-0.14** — renda bruta isolada tem fraca relação com inadimplência.

**Desbalanceamento:** a base possui ~78% de bons pagadores e ~22% de inadimplentes — tratado com **SMOTE** exclusivamente nos dados de treino.

---

## Decisões Técnicas

**Encoding:**
- `person_home_ownership`, `loan_intent`, `cb_person_default_on_file` → One-Hot Encoding (sem ordem natural).
- `loan_grade` → Codificação Ordinal (A=0 até G=6, pois há hierarquia de risco).

**Balanceamento:** SMOTE aplicado apenas em `X_treino`/`y_treino`, gerando 17.380 amostras por classe. Teste mantido intacto para refletir o mundo real.

**Escalonamento:** `StandardScaler` aplicado nos dados SMOTE para o KNN (`fit_transform`) e `transform` no teste. A Árvore de Decisão foi treinada **sem escalonamento**, pois nao interfere a escala.

---

## Resultados dos Modelos

### KNN — Comparativo de K

| K | Accuracy | F1 Classe 1 | Recall Classe 1 | Macro avg F1 |
|---|----------|-------------|-----------------|--------------|
| 3 | 0.85 | 0.66 | 0.68 | 0.78 |
| 5 | 0.86 | 0.67 | 0.69 | 0.79 |
| 7 | 0.87 | 0.69 | 0.69 | 0.80 |
| **9** | **0.87** | **0.69** | **0.69** | **0.81** |

**Melhor KNN: K=9** — maior precisão e macro avg F1. A partir de K=5 os ganhos são marginais.

### Árvore de Decisão — Comparativo de Profundidade

| Depth | Accuracy | F1 Classe 1 | Recall Classe 1 | Macro avg F1 |
|-------|----------|-------------|-----------------|--------------|
| 3 | 0.85 | 0.59 | 0.53 | 0.75 |
| 5 | 0.84 | 0.66 | 0.74 | 0.78 |
| **7** | **0.90** | **0.74** | **0.73** | **0.84** |
| None | 0.86 | 0.69 | 0.75 | 0.80 |

**Melhor Árvore: Depth=7** — melhor equilíbrio geral. Depth=None começa a decorar o treino (overfitting), reduzindo a qualidade no teste.

---

## Veredito Final

**Modelo escolhido para produção: Árvore de Decisão com `max_depth=7`**

| Critério | KNN-9 | Árvore Depth=7 | Vencedor |
|----------|-------|----------------|----------|
| Accuracy | 0.87 | **0.90** | Árvore |
| F1 Classe 1 | 0.69 | **0.74** | Árvore |
| Recall Classe 1 | 0.69 | **0.73** | Árvore |
| Falsos Negativos | 354 | **304** | Árvore |
| Interpretabilidade | ❌ | ✅ | Árvore |

A Árvore de Decisão vence em todas as métricas e, principalmente, comete **50 Falsos Negativos a menos** que o KNN — o erro mais custoso no contexto de crédito. Além disso, sua estrutura é **interpretável**: é possível visualizar exatamente quais variáveis e limiares levaram à decisão, o que é essencial para justificar recusas de crédito a clientes e atender exigências regulatórias do setor financeiro.

---

## Tecnologias Utilizadas

- Python 3.x
- pandas, numpy
- matplotlib, seaborn
- scikit-learn (KNeighborsClassifier, DecisionTreeClassifier, StandardScaler, train_test_split)
- imbalanced-learn (SMOTE)

---

## Como Executar

```bash
# 1. Clone o repositório
git clone <url-do-repositorio>

# 2. Instale as dependências
pip install pandas matplotlib seaborn scikit-learn imbalanced-learn

# 3. Execute o notebook
jupyter notebook Projeto_Final_Alexia_Colaco.ipynb
```

> O arquivo `credit_risk_dataset.csv` deve estar na mesma pasta do notebook.
