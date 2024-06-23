# Projeto de Tradução Automática de Sentenças (English to Portuguese)

Este projeto implementa um modelo de tradução automática de sentenças do inglês para o português utilizando redes neurais LSTM com Keras. O projeto envolve a preparação de dados, tokenização, construção e treinamento do modelo, além de uma interface para testar a tradução de novas sentenças.

## Arquivos do Projeto

### 1. `main.py`

Este script realiza as seguintes tarefas:

1. **Baixa e carrega os datasets**:
   - `sentences.csv` contendo sentenças em várias línguas.
   - `links.csv` contendo links entre sentenças que são traduções uma da outra.

2. **Filtra as sentenças**:
   - Seleciona sentenças em inglês e português.
   - Junta os datasets para obter pares de sentenças traduzidas.

3. **Pré-processamento**:
   - Adiciona tokens especiais `<start>` e `<end>` às sentenças em português.
   - Realiza a tokenização das sentenças.
   - Converte as sentenças em sequências de tokens e realiza padding para um comprimento fixo.

4. **Construção do modelo**:
   - Define um modelo seq2seq com LSTM para a tradução.
   - Compila e treina o modelo.

5. **Salva os artefatos**:
   - Salva os tokenizadores e o modelo treinado.

### 2. `test_model.py`

Este script realiza as seguintes tarefas:

1. **Carrega os tokenizadores e o modelo salvo**.
2. **Reconstrói os modelos de encoder e decoder** a partir do modelo salvo.
3. **Define funções auxiliares**:
   - `decode_sequence`: Para decodificar uma sequência de entrada e obter a tradução.
   - `prepare_input`: Para preparar o texto de entrada.
4. **Testa o modelo** com algumas frases de exemplo, imprimindo a tradução de cada uma.

## Link para os Artefatos

Todos os artefatos necessários para a execução do projeto (modelos, datasets, tokenizadores) estão disponíveis no Google Drive:

[Link para Google Drive](https://drive.google.com/drive/folders/18ylQWfwki8S3aR0QHT59hI-iMO1HDcJY?usp=sharing)

## Como Executar

### Passo 1: Preparar o Ambiente

1. Instalar as dependências necessárias:
   ```bash
   pip install pandas tensorflow scikit-learn

2. Baixar e preparar os datasets executando main.py:
    ```bash
    python main.py

### Passo2: Testar o Modelo
1. Executar test_model.py para testar o modelo com novas sentenças:
    ```bash
    python test_model.py

### Estrutura dos Arquivos
main.py: Script para preparação dos dados, construção e treinamento do modelo.
test_model.py: Script para testar o modelo treinado com novas sentenças.

### Conclusão
Este projeto demonstra a construção de um sistema de tradução automática utilizando LSTM. Ele abrange desde a preparação dos dados até a implementação e teste do modelo, fornecendo uma base sólida para aplicações mais avançadas de tradução automática.
