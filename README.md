# CIFAR-10: CNN própria e transfer learning

Infraestrutura compartilhada para os experimentos de classificação do CIFAR-10
do Trabalho Prático de Visão Computacional da UFJF. Este repositório oferece
configuração, preparação dos dados e uma divisão fixa de treino e validação.
As arquiteturas e os loops de treinamento serão fornecidos nas próximas etapas
do trabalho.

## Requisitos

- Python 3.10 ou superior
- PyTorch e torchvision
- NumPy, scikit-learn, Matplotlib e tqdm

## Instalação

No PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

No Linux ou macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Divisão fixa dos dados

Gere uma única vez a divisão estratificada dos 50.000 exemplos de treino:

```bash
python scripts/make_split.py
```

O script baixa o CIFAR-10 em `data/` e salva os índices dos 45.000 exemplos de
treino e dos 5.000 de validação em `splits/splits.npz`. Confere as 500 imagens
de validação por classe e a ausência de interseção entre os índices. O arquivo
existente não é substituído sem a opção explícita `--force`:

```bash
python scripts/make_split.py --force
```

O download tenta primeiro o endereço atual indicado pela página oficial do
CIFAR-10 e depois um espelho alternativo. Se ambos falharem por DNS ou falta de
conexão, baixe `cifar-10-python.tar.gz` manualmente para `data/` e rode o script
novamente; o `torchvision` verifica o checksum e extrai o arquivo.

Todos os integrantes devem compartilhar e usar esse mesmo arquivo de split. O
loader de treino e validação não retorna imagens do conjunto de teste.

## Carregar os dados

```python
from src.config import Config
from src.data import get_dataloaders

config = Config(experiment_name="meu_experimento", batch_size=128)
train_loader, val_loader = get_dataloaders(config)

imagens, rotulos = next(iter(train_loader))
print(imagens.shape)  # (batch, 3, 32, 32), salvo se resize estiver configurado
print(rotulos.shape)  # (batch,)
```

O aumento de dados é aplicado apenas ao treino. Validação (e teste, quando a
avaliação final for implementada) recebe apenas redimensionamento opcional,
conversão para tensor e normalização. Para imagens CIFAR-10 em RGB, a
normalização padrão usa média `(0.4914, 0.4822, 0.4465)` e desvio
`(0.2470, 0.2435, 0.2616)`. Modelos pré-treinados podem selecionar a
normalização ImageNet e um redimensionamento, por exemplo `resize=224`.

## Reprodutibilidade

Use `set_seed(config.seed)` antes de iniciar cada experimento. A semente padrão
é 42 e o split estratificado também usa 42. Não regenere nem altere o split
entre experimentos. O conjunto de teste deve permanecer separado até a seleção
das configurações finais pela validação.
