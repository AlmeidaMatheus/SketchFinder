# SketchFinder

O SketchFinder é uma aplicação desenvolvida em Python para busca de imagens utilizando fotografia, desenho livre ou descrição textual.

O projeto utiliza técnicas de Visão Computacional e Inteligência Artificial para representar imagens e textos através de vetores e encontrar os elementos mais semelhantes em um dataset.

## Funcionalidades

- Busca por upload de imagem
- Busca através de desenho feito com o mouse
- Busca através de descrição textual
- Comparação utilizando ResNet-18
- Comparação utilizando CLIP
- Comparação tradicional utilizando OpenCV
- Ranking das cinco imagens mais semelhantes
- Indexação prévia das imagens através de embeddings
- Preparação automática do dataset

## Tecnologias utilizadas

- Python
- Streamlit
- NumPy
- OpenCV
- Pillow
- PyTorch
- Torchvision
- ResNet-18
- CLIP
- Transformers
- Scikit-learn

## Dataset

O projeto utiliza um subconjunto do CIFAR-100 contendo 240 imagens distribuídas em 8 categorias, com 30 imagens por categoria.

Categorias utilizadas:

- bicycle
- bus
- motorcycle
- pickup_truck
- train
- house
- lion
- rocket

O conjunto completo do CIFAR-100 é necessário somente caso seja desejado reconstruir o dataset. Para a utilização normal, as 240 imagens selecionadas já acompanham o projeto.

## Embeddings

Para evitar que todas as imagens precisem passar novamente pelos modelos a cada pesquisa, suas características são previamente calculadas.

Os índices resultantes são armazenados nos arquivos:

- embeddings.pkl
- embeddings_clip.pkl

Durante uma pesquisa, somente a entrada informada pelo usuário precisa ter seu embedding calculado. O embedding é comparado com os embeddings previamente armazenados utilizando similaridade de cosseno.

## Execução

Criar ambiente virtual:

python -m venv .venv

Ativar o ambiente virtual no Windows:

.\.venv\Scripts\Activate.ps1

Instalar as dependências:

pip install -r requirements.txt

Executar a aplicação:

streamlit run app.py

## Métodos utilizados

### OpenCV

O OpenCV é utilizado para processamento tradicional de imagens, incluindo conversão para escala de cinza e detecção de bordas através do algoritmo Canny.

### ResNet-18

A ResNet-18 pré-treinada é utilizada como extratora de características. A camada final de classificação é removida e a representação interna da imagem é utilizada como embedding.

### CLIP

O CLIP permite representar imagens e textos em um espaço vetorial compartilhado. Isso possibilita realizar buscas utilizando imagens, desenhos e descrições textuais.

## Resultados observados

Nos testes realizados, o CLIP apresentou resultados consistentes na busca por texto e por fotografias.

Na busca através de desenhos, o desempenho varia de acordo com o objeto e com a forma como ele é representado. Objetos como casas e bicicletas apresentaram bons resultados, enquanto objetos visualmente semelhantes, como bicicletas e motocicletas, apresentaram maior possibilidade de confusão.

A ResNet-18 apresentou bons resultados na comparação entre fotografias, porém demonstrou maior dificuldade na comparação entre desenhos simples e fotografias.

## Limitações

- A qualidade do resultado depende da representação do objeto.
- Desenhos simplificados podem ser confundidos com objetos visualmente semelhantes.
- As imagens originais do CIFAR-100 possuem baixa resolução.
- Os modelos utilizados são pré-treinados e não foram especificamente treinados para busca entre desenhos e fotografias.