"""
Orquestrador do experimento.

Define a lista de cenários a executar e roda cada um chamando o framework.
No final, monta uma tabela comparativa única com todos os resultados.

Uso:
    python run_experiment.py

Saídas:
    results/<cenario>_*           : artefatos de cada cenário (ver trainer.py)
    results/comparativo.csv       : tabela final consolidada
    results/comparativo.md        : versão em markdown da tabela (pra colar no artigo)
"""

import os
import json
import time
import pandas as pd

from framework.trainer import run_scenario


# ----------------------------------------------------------------------
# Definição dos cenários do experimento.
#
# Cada cenário é (dataset, conjunto de técnicas). Adicionar/remover cenários
# aqui muda o experimento, sem mexer em nenhum outro arquivo.
# ----------------------------------------------------------------------
CENARIOS = [
    # CIFAR-10
    {'nome': 'cifar10_baseline',   'dataset': 'cifar10',       'tecnicas': []},
    {'nome': 'cifar10_rotation',   'dataset': 'cifar10',       'tecnicas': ['rotation']},
    {'nome': 'cifar10_flip',       'dataset': 'cifar10',       'tecnicas': ['flip']},
    {'nome': 'cifar10_brightness', 'dataset': 'cifar10',       'tecnicas': ['brightness']},
    {'nome': 'cifar10_blur',       'dataset': 'cifar10',       'tecnicas': ['blur']},
    {'nome': 'cifar10_combined',   'dataset': 'cifar10',       'tecnicas': ['rotation', 'flip', 'brightness', 'blur']},

    # Fashion-MNIST
    {'nome': 'fashion_baseline',   'dataset': 'fashion_mnist', 'tecnicas': []},
    {'nome': 'fashion_rotation',   'dataset': 'fashion_mnist', 'tecnicas': ['rotation']},
    {'nome': 'fashion_flip',       'dataset': 'fashion_mnist', 'tecnicas': ['flip']},
    {'nome': 'fashion_brightness', 'dataset': 'fashion_mnist', 'tecnicas': ['brightness']},
    {'nome': 'fashion_blur',       'dataset': 'fashion_mnist', 'tecnicas': ['blur']},
    {'nome': 'fashion_combined',   'dataset': 'fashion_mnist', 'tecnicas': ['rotation', 'flip', 'brightness', 'blur']},
]

EPOCHS = 25
BATCH_SIZE = 64
RESULTS_DIR = 'results'


def main():
    inicio = time.time()
    todas_metricas = []

    for i, cen in enumerate(CENARIOS, start=1):
        print(f"\n>>> Executando cenário {i}/{len(CENARIOS)}: {cen['nome']}")
        t0 = time.time()
        metricas = run_scenario(
            nome_cenario=cen['nome'],
            dataset=cen['dataset'],
            tecnicas=cen['tecnicas'],
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            results_dir=RESULTS_DIR,
        )
        metricas['tempo_segundos'] = round(time.time() - t0, 1)
        todas_metricas.append(metricas)

    # Tabela comparativa
    df = pd.DataFrame(todas_metricas)

    # Reordena colunas pra ficar legível
    colunas = [
        'cenario', 'dataset', 'tecnicas',
        'accuracy', 'f1_macro', 'f1_weighted',
        'test_loss', 'tamanho_treino', 'tempo_segundos'
    ]
    df = df[colunas]

    # Arredonda métricas pra leitura
    df['accuracy'] = df['accuracy'].round(4)
    df['f1_macro'] = df['f1_macro'].round(4)
    df['f1_weighted'] = df['f1_weighted'].round(4)
    df['test_loss'] = df['test_loss'].round(4)

    df.to_csv(os.path.join(RESULTS_DIR, 'comparativo.csv'), index=False)
    with open(os.path.join(RESULTS_DIR, 'comparativo.md'), 'w') as f:
        f.write(df.to_markdown(index=False))

    total = round(time.time() - inicio, 1)

    print("\n" + "="*60)
    print("EXPERIMENTO CONCLUÍDO")
    print("="*60)
    print(f"\nTempo total: {total}s ({total/60:.1f} min)")
    print(f"Resultados em: {RESULTS_DIR}/")
    print(f"  - comparativo.csv (tabela completa)")
    print(f"  - comparativo.md  (versão pro artigo)")
    print(f"  - {len(CENARIOS)} cenários × 5 artefatos cada\n")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
