# Projeto de Detecção de Imagens Reais e Falsas com FFT e ResNet

Este projeto demonstra como detectar imagens falsas usando a Transformada Rápida de Fourier (FFT) e um modelo de deep learning (ResNet). O pipeline inclui a preparação dos dados, treinamento do modelo e a implementação de uma API com FastAPI para fazer previsões.

## Estrutura do Projeto

- `fft_image_transform.py`: Script para transformar imagens em espectros de magnitude usando FFT.
- `model.py`: Script para treinar um modelo ResNet18 com os espectros de magnitude das imagens.
- `main.py`: Script FastAPI para receber imagens, aplicar FFT, e usar o modelo treinado para prever se a imagem é real ou falsa.
- `Dockerfile`: Arquivo para criar um contêiner Docker para a aplicação.

## Pré-requisitos

- Python 3.11
- OpenCV
- NumPy
- PyTorch
- FastAPI
- Uvicorn

## Instalação

### Configuração do Ambiente Python

1. Clone o repositório e navegue até o diretório do projeto:

    ```bash
    git clone https://github.com/seu-usuario/seu-repositorio.git
    cd seu-repositorio
    ```

2. Crie e ative um ambiente virtual:

    ```bash
    py -3.11 -m venv venv
    cd venv/Scripts
    activate
    cd ../..
    ```

3. Instale as dependências:

    ```bash
    pip install -r requirements.txt
    ```

## Preparação dos Dados

1. Crie uma pasta chamada `images` e adicione todas as imagens que serão usadas para construir os datasets de espectro.

2. Execute o script `fft_image_transform.py` para gerar os espectros de magnitude das imagens:

    ```bash
    python fft_image_transform.py
    ```

3. Separe os espectros gerados em pastas de treino, validação e teste dentro de um diretório chamado `dataset`, organizando-os nas subpastas `train`, `valid` e `test`. Cada uma dessas subpastas deve conter duas subpastas adicionais: `real` e `fake`. A estrutura do diretório deve ser como a seguir:

    ```
    dataset/
    ├── train/
    │   ├── real/
    │   └── fake/
    ├── valid/
    │   ├── real/
    │   └── fake/
    └── test/
        ├── real/
        └── fake/
    ```

    - Coloque os espectros de imagens reais na pasta `real` correspondente.
    - Coloque os espectros de imagens falsas na pasta `fake` correspondente.

## Treinamento do Modelo

1. Use o script `model.py` para treinar um modelo ResNet18 com os espectros de magnitude das imagens:

    ```bash
    python model.py
    ```

2. O modelo treinado será salvo como `model_resnet18.pth`, após a conclusão da iteração da Época como no exemplo abaixo.

    ```
    Epoch 18/20
    ----------
    train Loss: 0.1709 Acc: 0.9369 Precision: 0.9380 Recall: 0.9369 F1: 0.9369
    valid Loss: 0.1107 Acc: 0.9649 Precision: 0.9672 Recall: 0.9649 F1: 0.9648

    Epoch 19/20
    ----------
    train Loss: 0.1852 Acc: 0.9272 Precision: 0.9272 Recall: 0.9272 F1: 0.9272
    valid Loss: 0.0916 Acc: 0.9649 Precision: 0.9672 Recall: 0.9649 F1: 0.9648

    Epoch 20/20
    ----------
    train Loss: 0.1819 Acc: 0.9320 Precision: 0.9320 Recall: 0.9320 F1: 0.9320
    valid Loss: 0.0814 Acc: 0.9825 Precision: 0.9830 Recall: 0.9825 F1: 0.9824

    Training complete in 9m 46s
    ```

    *Verifique o `model_resnet18.pth` criado na raiz do projeto.

## Implementação da API com FastAPI

1. Use o script `main.py` para implementar uma API que recebe imagens, aplica FFT, e usa o modelo treinado para prever se a imagem é real ou falsa:

    ```bash
    python main.py
    ```

2. A API estará disponível em `http://localhost:8000`.

## Executando com Docker

1. Construa a imagem Docker:

    ```bash
    docker build -t my-fastapi-app .
    ```

2. Execute o contêiner Docker:

    ```bash
    docker run -d --name my-fastapi-app -p 8000:8000 my-fastapi-app
    ```

3. A API estará disponível em `http://localhost:8000`.

## Endpoints da API

- `POST /predict/`: Recebe uma imagem, aplica FFT e usa o modelo treinado para prever se a imagem é real ou falsa.

### Exemplo de Uso

1. Enviar uma imagem para o endpoint `/predict/`:

    ```bash
    curl -X 'POST' \
      'http://localhost:8000/predict/' \
      -H 'accept: application/json' \
      -H 'Content-Type: multipart/form-data' \
      -F 'file=@path/to/your/image.jpg'
    ```

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues e pull requests.


---

Sinta-se à vontade para ajustar e expandir conforme necessário. Se houver algo mais que você gostaria de adicionar ou modificar, avise-me!
