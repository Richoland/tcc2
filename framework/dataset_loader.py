"""
Carregador de datasets do framework.

Abstrai qual dataset usar e devolve um dicionário 'info' com os metadados
necessários para que o resto do framework (modelo, treino, avaliação) se
adapte sozinho ao dataset escolhido.
"""

from tensorflow.keras.datasets import cifar10, fashion_mnist
import numpy as np


CIFAR10_CLASSES = [
    'Avião', 'Automóvel', 'Pássaro', 'Gato', 'Veado',
    'Cachorro', 'Sapo', 'Cavalo', 'Navio', 'Caminhão'
]

FASHION_CLASSES = [
    'Camiseta', 'Calça', 'Pulôver', 'Vestido', 'Casaco',
    'Sandália', 'Camisa', 'Tênis', 'Bolsa', 'Bota'
]


def load_dataset(nome):
    """
    Carrega o dataset pelo nome e devolve treino, teste e metadados.

    Parâmetros
    ----------
    nome : str
        'cifar10' ou 'fashion_mnist'

    Retorna
    -------
    x_train, y_train, x_test, y_test, info
        Imagens e rótulos de treino/teste + dicionário com metadados.
    """
    if nome == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        info = {
            'input_shape': (32, 32, 3),
            'num_classes': 10,
            'class_names': CIFAR10_CLASSES,
            'name': 'cifar10',
        }

    elif nome == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        # Fashion-MNIST vem com shape (N, 28, 28); a CNN precisa de (N, 28, 28, 1).
        # Adiciona o canal explícito no final.
        x_train = x_train[..., np.newaxis]
        x_test = x_test[..., np.newaxis]
        info = {
            'input_shape': (28, 28, 1),
            'num_classes': 10,
            'class_names': FASHION_CLASSES,
            'name': 'fashion_mnist',
        }

    else:
        raise ValueError(
            f"Dataset desconhecido: '{nome}'. "
            f"Use 'cifar10' ou 'fashion_mnist'."
        )

    return x_train, y_train, x_test, y_test, info
