# CIFAR-10: CNN própria e transfer learning

Infraestrutura compartilhada para os experimentos de classificação do CIFAR-10
do Trabalho Prático de Visão Computacional da UFJF. O repositório oferece
configuração, preparação dos dados, treinamento e métricas de validação. Cada
integrante define seu modelo e usa a mesma divisão fixa de treino e validação.

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

No Windows, `pip install -r requirements.txt` instala o PyTorch somente para
CPU. Para usar uma GPU NVIDIA, instale primeiro o PyTorch com o comando indicado
em [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)
e depois instale os demais pacotes do projeto. No Google Colab e no Kaggle, o
PyTorch já vem instalado.

Execute sempre os comandos a partir da raiz do repositório. Os imports
`from src...` e os caminhos relativos dependem desse diretório.

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

## Treinar e validar um modelo

Defina o modelo como uma subclasse de `torch.nn.Module` que retorne 10 logits por imagem. Configure os hiperparâmetros e execute `run_experiment`:

```python
from torch import nn
from src.config import Config
from src.experiment import run_experiment
from src.utils import set_seed

class MeuModelo(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, 10)

    def forward(self, imagens):
        return self.classifier(self.features(imagens).flatten(start_dim=1))

config = Config(
    experiment_name='minha_cnn', seed=42, optimizer='sgd', lr=0.01,
    momentum=0.9, weight_decay=0.0001, batch_size=128, epochs=20,
    scheduler='cosine', device='auto',
)
set_seed(config.seed)
modelo = MeuModelo()
resultado = run_experiment(modelo, config)
print(resultado['validation']['accuracy'])
```

Chame `set_seed(config.seed)` antes de instanciar o modelo para que os pesos
iniciais também dependam da semente.

O JSON fica em `results/runs/<experiment_name>.json` e o checkpoint da melhor acurácia de validação fica em `checkpoints/<experiment_name>_best.pt`. O scheduler `none` mantém a taxa de aprendizado fixa; `step` usa `StepLR` com `step_size=max(1, epochs // 3)` e `gamma=0.1`; `cosine` usa `CosineAnnealingLR` com `T_max=epochs`. Este fluxo usa apenas treino e validação e nunca carrega o conjunto de teste.

Para conferir o pipeline com subconjuntos pequenos do CIFAR-10:

```powershell
python scripts/smoke_test.py --fast
```

## Nomes de experimentos

O valor de `experiment_name` define o nome do JSON em
`results/runs/<nome>.json` e do checkpoint em `checkpoints/<nome>_best.pt`.
Repetir um nome sobrescreve os resultados anteriores. Para evitar isso, cada
integrante deve usar um prefixo com seu número, por exemplo `p2_baseline`,
`p3_batchnorm`, `p4_resnet18_frozen` e `p5_mobilenetv2_finetune`.

## Observação para Windows

Se `num_workers` for maior que zero, o `DataLoader` usa multiprocessamento. Em
scripts de treinamento no Windows, coloque o código principal dentro do bloco
`if __name__ == "__main__":` para evitar travamentos ou erros:

```python
def main():
    config = Config(experiment_name="p2_baseline")
    set_seed(config.seed)
    modelo = MeuModelo()
    run_experiment(modelo, config)

if __name__ == "__main__":
    main()
```
