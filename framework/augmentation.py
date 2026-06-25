"""
Pipeline de aumento de dados configurável.

Permite construir o pipeline a partir de uma lista de técnicas escolhidas,
em vez de ter as técnicas fixas no código. É o que torna o framework
configurável: cada cenário do experimento passa uma lista diferente.

As técnicas disponíveis:
  - 'rotation'   : rotação aleatória em ±15°
  - 'flip'       : espelhamento horizontal
  - 'brightness' : variação aleatória de brilho/contraste
  - 'blur'       : desfoque gaussiano leve
"""

import albumentations as A
import numpy as np


# Cada técnica é uma "fábrica" que devolve a transformação do Albumentations.
# Manter num dicionário facilita adicionar novas técnicas depois sem mexer
# em mais nada do framework.
TECNICAS_DISPONIVEIS = {
    'rotation':   lambda: A.Rotate(limit=15, p=0.5),
    'flip':       lambda: A.HorizontalFlip(p=0.5),
    'brightness': lambda: A.RandomBrightnessContrast(p=0.5),
    'blur':       lambda: A.GaussianBlur(blur_limit=(3, 5), p=0.5),
}


def build_pipeline(tecnicas):
    """
    Monta um pipeline do Albumentations a partir da lista de técnicas.

    Parâmetros
    ----------
    tecnicas : list[str]
        Lista de nomes das técnicas a aplicar. Lista vazia = sem augmentation.

    Retorna
    -------
    Compose | None
        Pipeline pronto para aplicar, ou None se a lista estiver vazia.
    """
    if not tecnicas:
        return None

    transformacoes = []
    for nome in tecnicas:
        if nome not in TECNICAS_DISPONIVEIS:
            raise ValueError(
                f"Técnica desconhecida: '{nome}'. "
                f"Disponíveis: {list(TECNICAS_DISPONIVEIS.keys())}"
            )
        transformacoes.append(TECNICAS_DISPONIVEIS[nome]())

    return A.Compose(transformacoes)


def apply_augmentation(x_train, y_train, tecnicas, concatenar=True):
    """
    Aplica o aumento de dados ao conjunto de treino.

    Parâmetros
    ----------
    x_train, y_train : ndarray
        Imagens e rótulos de treino originais.
    tecnicas : list[str]
        Lista de técnicas a aplicar. Vazia = devolve os dados originais.
    concatenar : bool, default True
        Se True, ACRESCENTA as versões aumentadas às originais (conjunto
        dobra de tamanho — comportamento padrão de augmentation).
        Se False, SUBSTITUI as imagens originais pelas versões transformadas
        (comportamento do código antigo, mantém o tamanho).

    Retorna
    -------
    x_aug, y_aug : ndarray
        Conjunto de treino após o aumento.
    """
    pipeline = build_pipeline(tecnicas)

    # Cenário baseline: nenhuma técnica, devolve original.
    if pipeline is None:
        return x_train, y_train

    # Aplica o pipeline imagem por imagem.
    augmented = np.array([pipeline(image=img)['image'] for img in x_train])

    if concatenar:
        # Junta originais + aumentadas (dobra o conjunto).
        x_aug = np.concatenate([x_train, augmented], axis=0)
        y_aug = np.concatenate([y_train, y_train], axis=0)
        return x_aug, y_aug
    else:
        # Só substitui (mantém o tamanho original).
        return augmented, y_train
