"""
Executor de cenário do framework.

Um "cenário" é uma combinação de (dataset, técnicas de augmentation). Esta
função executa o ciclo completo: carrega dados, aplica augmentation, treina,
avalia e salva resultados em arquivos identificados pelo nome do cenário.

Saídas geradas por cenário (na pasta indicada por results_dir):
  - <nome>_model.keras           : modelo treinado
  - <nome>_metrics.json          : métricas (acurácia, F1 macro/weighted, loss)
  - <nome>_classification.txt    : relatório de classificação por classe
  - <nome>_confusion_matrix.png  : matriz de confusão com nomes das classes
  - <nome>_history.csv           : histórico de treino (loss/acc por época)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # WSL: sem janelas interativas, só salva arquivos
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score
)

from framework.dataset_loader import load_dataset
from framework.augmentation import apply_augmentation
from framework.model import build_model


def run_scenario(
    nome_cenario,
    dataset,
    tecnicas,
    epochs=25,
    batch_size=64,
    concatenar=True,
    results_dir='results',
):
    """
    Executa um cenário completo do experimento.

    Parâmetros
    ----------
    nome_cenario : str
        Identificador do cenário, ex: 'cifar10_baseline', 'cifar10_rotation',
        'fashion_mnist_combined'. Usado pra nomear os arquivos de saída.
    dataset : str
        Nome do dataset: 'cifar10' ou 'fashion_mnist'.
    tecnicas : list[str]
        Lista de técnicas de augmentation. Vazia = baseline.
    epochs, batch_size : int
        Hiperparâmetros do treino.
    concatenar : bool
        Se True, augmentation acrescenta às originais; se False, substitui.
    results_dir : str
        Pasta onde salvar os arquivos de saída.

    Retorna
    -------
    dict
        Métricas do cenário (também salvas em <nome>_metrics.json).
    """
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Cenário: {nome_cenario}")
    print(f"Dataset: {dataset} | Técnicas: {tecnicas or 'nenhuma (baseline)'}")
    print(f"{'='*60}\n")

    # 1) Dataset
    x_train, y_train, x_test, y_test, info = load_dataset(dataset)

    # 2) Augmentation (ANTES da normalização, porque Albumentations espera
    #    valores 0-255 em uint8 para algumas operações)
    x_train, y_train = apply_augmentation(
        x_train, y_train, tecnicas, concatenar=concatenar
    )
    print(f"Tamanho do treino após augmentation: {len(x_train)}")

    # 3) Normalização
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # 4) Modelo
    model = build_model(info)

    # 5) Treino
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        verbose=2,  # uma linha por época (menos poluído que verbose=1)
    )

    # 6) Avaliação no conjunto de teste
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    y_true = y_test.flatten()

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    metrics = {
        'cenario': nome_cenario,
        'dataset': dataset,
        'tecnicas': tecnicas,
        'epochs': epochs,
        'batch_size': batch_size,
        'concatenar': concatenar,
        'tamanho_treino': len(x_train),
        'accuracy': float(acc),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'test_loss': float(test_loss),
    }

    # 7) Salvar artefatos

    # 7.1) Modelo
    model.save(os.path.join(results_dir, f'{nome_cenario}_model.keras'))

    # 7.2) Métricas (json)
    with open(os.path.join(results_dir, f'{nome_cenario}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # 7.3) Relatório de classificação por classe
    report = classification_report(
        y_true, y_pred, target_names=info['class_names']
    )
    with open(os.path.join(results_dir, f'{nome_cenario}_classification.txt'), 'w') as f:
        f.write(report)

    # 7.4) Histórico de treino (loss/acc por época)
    pd.DataFrame(history.history).to_csv(
        os.path.join(results_dir, f'{nome_cenario}_history.csv'), index=False
    )

    # 7.5) Matriz de confusão com nomes das classes
    cm = confusion_matrix(y_true, y_pred)
    _plot_confusion_matrix(
        cm, info['class_names'], nome_cenario,
        os.path.join(results_dir, f'{nome_cenario}_confusion_matrix.png')
    )

    print(f"\n>>> {nome_cenario}: accuracy={acc:.4f} | f1_macro={f1_macro:.4f}")
    print(f"Artefatos salvos em: {results_dir}/{nome_cenario}_*")

    return metrics


def _plot_confusion_matrix(cm, class_names, titulo, caminho_saida):
    """Plota e salva a matriz de confusão com nomes de classe."""
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(8.2, 6.6), dpi=150)
    im = ax.imshow(cm, cmap='Greens')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel('Classe Predita', fontsize=11, fontweight='bold')
    ax.set_ylabel('Classe Verdadeira', fontsize=11, fontweight='bold')
    ax.set_title(f'Matriz de Confusão — {titulo}', fontsize=12, fontweight='bold', pad=12)

    thr = cm.max() * 0.5
    for i in range(n):
        for j in range(n):
            v = cm[i, j]
            ax.text(
                j, i, str(v),
                ha='center', va='center',
                color='white' if v > thr else '#333',
                fontsize=8,
                fontweight='bold' if i == j else 'normal',
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=150, bbox_inches='tight')
    plt.close(fig)
