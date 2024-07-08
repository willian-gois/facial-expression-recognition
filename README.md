# Reconhecimento de Expressões Faciais em Tempo Real

Projeto desenvolvido para a disciplina de Tópicos Avançados no curso de Análise e Desenvolvimento de Sistemas no IFRS Campus Farroupilha.

O projeto é um sistema de reconhecimento de expressões faciais em tempo real utilizando uma rede neural convolucional (CNN) treinada através do train.ipynb. O objetivo é detectar e classificar expressões faciais como felicidade, tristeza, raiva, entre outras, através da câmera do computador.

## Pré-requisitos

Para executar este projeto, é necessário ter instalado:

- Python >= 3.8
- Bibliotecas Python necessárias (listadas no arquivo `requirements.txt`)

## Instalação

1. Clone este repositório:
```bash
git clone https://github.com/willian-gois/facial-expression-recognition.git
cd facial-expression-recognition.
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Uso

1. Execute o script principal:
```bash
python main.py
```

2. A câmera será ativada automaticamente, e as expressões faciais detectadas serão exibidas na interface gráfica.

## Treinamento da Rede Neural

O modelo da rede neural e seus pesos já estão versionados neste repositório (./data), mas caso desejar treiná-la usando outro conjunto de dados ou ajustar o modelo, recomendamos usar GPUs disponíveis no Google Collab para acelerar o processo. O dataset original utilizado neste projeto pode ser encontrado [aqui](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/overview).

Para treinar a rede, você pode adaptar o train.ipynb fornecido neste repositório e realizar o treinamento no Google Collab, carregando o conjunto de dados e ajustando os parâmetros conforme necessário.