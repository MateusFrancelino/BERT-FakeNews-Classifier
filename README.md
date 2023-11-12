# Instruções para Execução do Projeto

## 1. Base.ipynb

- **Arquivo Base:** `Base.ipynb`
- **Caminho do CSV:** `"C:\TCC\Base\merged_data.csv"`

Antes de executar o código, siga as etapas abaixo:

    a. Extrair o arquivo `merged_data.zip` que está neste repositório.
    b. Substituir o caminho do arquivo em `CSV_PATH` no código.
    
**Dependências:**
- pandas
- scikit-learn

---

## 2. BERTimbau.ipynb

- **Arquivo BERTimbau:** `BERTimbau.ipynb`

Antes de executar o código, siga as etapas abaixo:

    a. Substituir os caminhos dos CSV em `train_data`, `validade_data`, e `test_data`.
    b. Substituir o local onde o modelo é salvo em `torch.save`.

**Dependências:**
- pandas
- scikit-learn
- transformers
- torch

---

## 3. AvaliarModelos.ipynb

- **Arquivo AvaliarModelos:** `AvaliarModelos.ipynb`

Antes de executar o código, siga as etapas abaixo:

    a. Substituir os caminhos para o modelo e tokenizador treinado em `caminhos`.
    b. Substituir o caminho para o arquivo CSV de teste em `caminho_csv`.
    c. Substituir o local de armazenamento dos resultados em `resultados_csv`.

**Dependências:**
- torch
- pandas
- transformers
- scikit-learn

---

**Nota:** A base utilizada neste projeto foi a Fake.Br Corpus disponível em [https://github.com/roneysco/Fake.br-Corpus](https://github.com/roneysco/Fake.br-Corpus).
