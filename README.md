# AI Draw Battle

AI Draw Battle é uma aplicação desenvolvida em Python que compara dois modelos de Inteligência Artificial na tentativa de identificar, em tempo real, um objeto desenhado pelo usuário.

O usuário informa previamente o que será desenhado e, durante o desenho, CLIP e ResNet-18 realizam sucessivas tentativas de reconhecimento. O primeiro modelo que acertar a resposta vence a rodada.

## Funcionalidades

- Desenho livre utilizando o mouse
- Reconhecimento do desenho em tempo real
- Competição entre CLIP e ResNet-18
- Histórico de palpites durante a rodada
- Encerramento automático quando um modelo acerta
- Encerramento manual da rodada
- Exibição da resposta correta ao final
- Análise do quanto cada modelo estava relacionado à resposta correta

## Tecnologias utilizadas

- Python
- Streamlit
- Streamlit Drawable Canvas
- NumPy
- Pillow
- PyTorch
- Torchvision
- ResNet-18
- Transformers
- CLIP

## Modelos utilizados

### ResNet-18

A ResNet-18 é utilizada com pesos pré-treinados no ImageNet.

Durante o jogo, o desenho é processado pela rede e classificado entre as categorias conhecidas pelo modelo.

A ResNet-18 realiza uma classificação fechada, ou seja, sua resposta precisa estar entre as categorias presentes em sua camada de classificação.

### CLIP

O CLIP é um modelo multimodal capaz de criar representações para imagens e textos em um espaço vetorial compartilhado.

Durante o início da rodada, a resposta informada pelo usuário e as categorias utilizadas como alternativas são transformadas em embeddings textuais.

Enquanto o usuário desenha, o desenho é transformado em um embedding visual e comparado com os embeddings dos textos.

O candidato com maior similaridade é utilizado como palpite do CLIP.

Essa abordagem permite que o CLIP avalie conceitos informados em tempo de execução, mesmo que eles não façam parte de uma classificação fechada como ocorre na ResNet-18.

## Funcionamento

O usuário informa inicialmente o objeto que será desenhado.

Exemplo:

house

A resposta fica escondida e uma nova rodada é iniciada.

Durante o desenho, cada alteração relevante no canvas é enviada aos dois modelos.

Fluxo simplificado:

Desenho
  |
  +-- ResNet-18
  |      |
  |      +-- classificação
  |      +-- palpite
  |
  +-- CLIP
         |
         +-- embedding visual
         +-- comparação com embeddings textuais
         +-- palpite

Cada modelo realiza somente um palpite por atualização do desenho.

Os palpites são armazenados e exibidos no histórico da rodada.

Quando um dos modelos acerta a resposta, a rodada é encerrada automaticamente.

O usuário também pode encerrar a rodada manualmente.

## CLIP e classificação zero-shot

O CLIP permite utilizar a resposta informada pelo usuário como um possível candidato durante a classificação.

Por exemplo, se o usuário informar:

dragon

o termo pode ser transformado em um embedding textual e comparado com o desenho mesmo sem existir como uma classe específica da ResNet-18 utilizada pela aplicação.

Isso permite demonstrar a diferença entre uma classificação fechada e uma abordagem zero-shot.

## Histórico

Durante uma rodada, cada atualização analisada gera:

- um palpite do CLIP
- um palpite da ResNet-18
- o score correspondente de cada modelo

Exemplo:

1
CLIP: envelope
ResNet-18: envelope

2
CLIP: church
ResNet-18: barn

3
CLIP: house
ResNet-18: barn

Nesse exemplo, o CLIP vence a rodada ao identificar corretamente house.

## Encerramento da rodada

Caso nenhum modelo acerte, o usuário pode encerrar a rodada manualmente.

Ao final são apresentados:

- resposta correta
- último palpite do CLIP
- último palpite da ResNet-18
- histórico completo de palpites
- similaridade do CLIP com a resposta correta
- probabilidade atribuída pela ResNet-18 à resposta, quando essa resposta existir entre suas classes

É importante observar que os valores apresentados pelos dois modelos possuem significados diferentes.

Na ResNet-18 é apresentada uma probabilidade obtida a partir da distribuição de classificação.

No CLIP é apresentada uma similaridade entre as representações do desenho e do texto. Essa similaridade não deve ser interpretada como uma probabilidade de classificação.

## Execução

### Criar ambiente virtual

python -m venv .venv

### Ativar ambiente virtual no Windows

.\.venv\Scripts\Activate.ps1

### Instalar dependências

pip install -r requirements.txt

### Executar

streamlit run app.py

## Estrutura

AI-Draw-Battle/
|
|-- app.py
|-- requirements.txt
|-- README.md
`-- .gitignore

## Limitações

- A ResNet-18 está limitada às categorias da classificação para a qual seus pesos foram treinados.
- O CLIP precisa receber candidatos textuais para escolher seu palpite e não gera palavras livremente.
- O reconhecimento depende da qualidade e do estágio atual do desenho.
- Desenhos muito simples podem ser associados a objetos visualmente semelhantes.
- Os scores produzidos pelo CLIP e pela ResNet-18 não representam a mesma métrica e não devem ser comparados diretamente.
- O desempenho pode variar dependendo da execução em CPU ou GPU.