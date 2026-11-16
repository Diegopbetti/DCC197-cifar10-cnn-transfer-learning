# Pipeline de treino, validação e resultados

## Contexto

O projeto já contém configuração de experimentos, carregadores de dados com um
split fixo, e fixação de sementes. A próxima etapa precisa permitir que os
integrantes treinem seus próprios `torch.nn.Module` usando o mesmo pipeline,
sem incluir as arquiteturas do trabalho na infraestrutura compartilhada.

## Objetivos

- Treinar e validar qualquer modelo PyTorch que produza logits para as 10
  classes do CIFAR-10.
- Selecionar e salvar o checkpoint de maior acurácia de validação.
- Registrar métricas, tempos, parâmetros, ambiente e matriz de confusão em um
  JSON por experimento.
- Fornecer uma chamada simples, `run_experiment(model, config)`, e um script de
  verificação rápida que percorra o fluxo com dados reduzidos.
- Manter o conjunto de teste fora do fluxo de desenvolvimento.

## Fora de escopo

- Implementar a CNN baseline, suas variações ou arquiteturas pré-treinadas.
- Avaliar ou carregar o conjunto de teste.
- Gerar gráficos ou tabelas LaTeX; isso pertence à etapa posterior.
- Alterar a regra, a semente ou os índices do split versionado.

## Organização e responsabilidades

- `src/train.py`: `train_model(model, config, train_loader, val_loader)` cria
  perda, otimizador e scheduler, executa treino e validação por época, mantém o
  histórico e salva o melhor `state_dict` em
  `checkpoints/<experiment_name>_best.pt`.
- `src/evaluate.py`: `evaluate(model, loader, device)` calcula perda média,
  acurácia, predições e rótulos. `confusion_matrix_from(y_true, y_pred)` calcula
  a matriz 10 x 10 usando scikit-learn.
- `src/experiment.py`: `run_experiment(model, config, *, train_loader=None,
  val_loader=None)` coordena a semente, os loaders, o treino, a avaliação final
  na validação, as medições e a gravação do JSON. Os loaders opcionais servem
  apenas para permitir que o script de verificação injete subconjuntos; devem
  ser fornecidos juntos ou omitidos juntos.
- `src/utils.py`: acrescenta contagem de parâmetros, tempo de inferência com
  aquecimento e sincronização CUDA, informações de hardware e gravação JSON.
- `scripts/smoke_test.py`: define `SimpleSmokeCNN`, identificada como modelo
  exclusivo para verificar a infraestrutura, e chama `run_experiment` com
  subconjuntos pequenos. A opção `--fast` reduz o tamanho e executa uma época.
- `README.md`: acrescenta um exemplo curto de criação de modelo pelo colega e
  chamada de `run_experiment`.

## Fluxo de execução

1. `run_experiment` fixa as sementes e resolve `device="auto"` para CUDA, se
   disponível, ou CPU.
2. Se os loaders não forem fornecidos, carrega apenas treino e validação por
   `get_dataloaders(config)`. Não importa nem chama `get_test_loader`.
3. `train_model` treina por `config.epochs`, calcula métricas de treino e
   validação em cada época e salva os pesos quando a acurácia de validação
   supera o melhor valor anterior.
4. O treinador retorna histórico, melhor acurácia e época, caminho do checkpoint
   e tempos por época e total. O tempo total sincroniza CUDA quando aplicável.
5. `run_experiment` restaura o melhor checkpoint, calcula perda, acurácia e
   matriz de confusão finais na validação, mede inferência nesse loader,
   registra parâmetros e hardware e grava
   `results/runs/<experiment_name>.json`.

## Regras de treino

- A perda é `torch.nn.CrossEntropyLoss`.
- SGD e Adam são construídos a partir de `config.optimizer`; ambos recebem
  somente parâmetros com `requires_grad=True`. Um modelo sem parâmetros
  treináveis resulta em erro explicativo.
- `scheduler="none"` não cria scheduler. `step` usa `StepLR`, com
  `step_size=max(1, epochs // 3)` e `gamma=0.1`. `cosine` usa
  `CosineAnnealingLR` com `T_max=epochs`. Essas escolhas padrão são documentadas
  no README; os valores configuráveis existentes permanecem em `Config`.
- Perdas são agregadas ponderando pelo tamanho do batch; acurácia é o número de
  acertos dividido pelo total de exemplos.
- O histórico contém `train_loss`, `train_acc`, `val_loss`, `val_acc`, `lr` e
  `epoch_time_seconds` para cada época. `tqdm` exibe o progresso.
- A melhor época usa a maior `val_acc`; em empate, conserva-se o primeiro
  checkpoint que atingiu aquele valor.
- O cronômetro de inferência aquece o modelo antes de medir e sincroniza CUDA
  antes e depois das medições na GPU. O valor retornado é segundos por imagem.

## Dados do JSON

O JSON contém a configuração serializada, histórico por época, melhor
`val_acc`, melhor época, métricas finais da validação (incluindo a matriz de
confusão), contagens de parâmetros, tempos de treino e inferência e informações
do ambiente (Python, PyTorch, CUDA e dispositivo). Predições e rótulos podem
ser retornados por `evaluate`, mas não são duplicados no JSON; a matriz de
confusão é o resumo persistido nesta etapa.

## Erros e limites

- Erros já emitidos por ausência ou formato inválido do split permanecem claros
  e propagam para o chamador.
- O pipeline rejeita loaders opcionais incompletos, ausência de parâmetros
  treináveis e saídas incompatíveis com os rótulos.
- Falhas não produzem um JSON que pareça representar um experimento concluído.
- O pipeline não consulta teste em nenhum caminho de execução.

## Critérios de aceitação

- `python scripts/smoke_test.py --fast` treina a `SimpleSmokeCNN` em um
  subconjunto pequeno por uma época e confirma que checkpoint e JSON foram
  gravados.
- `run_experiment(model, config)` funciona com os loaders completos e gera um
  único JSON no caminho previsto.
- O treino respeita as opções da configuração, atualiza corretamente o melhor
  checkpoint e retorna as métricas descritas.
- Avaliação e matriz de confusão usam apenas os loaders recebidos; não há acesso
  implícito ao teste.
- O README mostra o uso do pipeline com um modelo definido pelo colega.
