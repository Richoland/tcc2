# Framework para Avaliação Sistemática de Técnicas de Aumento de Dados em Imagens

Trabalho de Conclusão de Curso II — Bacharelado em Ciência da Computação, IFSul Campus Passo Fundo, 2026.

**Autor:** Ricardo Ilan Dall'Agnol
**Orientador:** Prof. Dr. João Mário Lopes Brezolin

## Descrição

Framework modular para avaliação sistemática e controlada de técnicas de aumento de dados em problemas de classificação de imagens. A ferramenta executa experimentos comparativos mantendo constantes a arquitetura da rede, os hiperparâmetros de treinamento e o conjunto de dados, variando apenas a configuração de aumento, o que permite isolar o efeito de cada técnica avaliada.

## Arquitetura

A biblioteca está organizada em cinco módulos:

- framework/dataset_loader.py - carrega CIFAR-10 ou Fashion-MNIST e fornece os metadados do dataset
- framework/augmentation.py - constrói o pipeline de aumento a partir da lista de técnicas (Albumentations)
- framework/model.py - constrói a CNN adaptada ao formato do dataset
- framework/trainer.py - executa um cenário (treina, avalia, salva artefatos)
- run_experiment.py - orquestrador: define a lista de cenários e consolida os resultados

## Como executar

Criar ambiente virtual com Python 3.12, instalar dependências e rodar:

    python3.12 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    python run_experiment.py

Os resultados são gerados em results/: tabela consolidada (comparativo.csv, comparativo.md), métricas por cenário (*_metrics.json), relatórios de classificação (*_classification.txt), históricos de treino (*_history.csv) e matrizes de confusão (*_confusion_matrix.png). Os modelos treinados (*.keras) também são salvos localmente, mas não são versionados.

## Cenários executados no TCC

12 cenários (2 datasets x 6 configurações): baseline sem aumento, quatro técnicas isoladas (rotação, espelhamento horizontal, brilho/contraste e desfoque gaussiano) e a combinação das quatro. Aplicados a CIFAR-10 e Fashion-MNIST.

## Licença

MIT (ver arquivo LICENSE).
