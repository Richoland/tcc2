"""
CNN do framework, adaptável ao dataset.

A arquitetura é a mesma da versão original (3 blocos convolucionais), mas
agora a camada de entrada e a saída são montadas a partir do dicionário
'info' do dataset. Assim o mesmo código serve para CIFAR-10 (32x32x3, 10
classes) e Fashion-MNIST (28x28x1, 10 classes).
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, Input
)
from tensorflow.keras.optimizers import Adam


def build_model(info):
    """
    Constrói a CNN adaptada ao dataset descrito em 'info'.

    Parâmetros
    ----------
    info : dict
        Dicionário do dataset_loader. Usa 'input_shape' e 'num_classes'.

    Retorna
    -------
    model : Sequential
        Rede compilada, pronta para treinar.
    """
    input_shape = info['input_shape']
    num_classes = info['num_classes']

    model = Sequential([
        # Camada de entrada explícita — adapta-se ao dataset.
        Input(shape=input_shape),

        # Bloco 1
        Conv2D(64, (3, 3), activation='swish'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Bloco 2
        Conv2D(128, (3, 3), activation='swish'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Bloco 3
        Conv2D(256, (3, 3), activation='swish'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Camadas densas finais
        Flatten(),
        Dense(512, activation='swish'),
        BatchNormalization(),
        Dropout(0.5),

        # Saída adapta-se ao número de classes do dataset.
        Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model
