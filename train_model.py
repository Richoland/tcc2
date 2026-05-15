from augmentation_pipeline import get_augmentation_pipeline
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import numpy as np

# Carregar CIFAR-10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Obter pipeline para treinamento
augmentation_pipeline = get_augmentation_pipeline(for_training=False)

# Aplicar aumentação aos dados de treinamento
augmented_images = []
augmented_labels = []

for img, label in zip(train_images, train_labels):
    augmented = augmentation_pipeline(image=img)
    augmented_images.append(augmented['image'])
    augmented_labels.append(label)

# Converter para arrays do NumPy
augmented_images = np.array(augmented_images)
augmented_labels = np.array(augmented_labels)

# Normalizar os dados
augmented_images = augmented_images / 255.0
test_images = test_images / 255.0

# Criar o modelo aprimorado
model = Sequential([
    # Bloco 1
    Conv2D(64, (3, 3), activation='swish', input_shape=(32, 32, 3)),
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

    # Camadas densas
    Flatten(),
    Dense(512, activation='swish'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')  # 10 classes para CIFAR-10
])

# Compilar o modelo com um otimizador ajustado
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Treinar o modelo
history = model.fit(
    augmented_images, augmented_labels,
    epochs=25,  # Aumente o número de épocas para melhor convergência
    batch_size=64,  # Ajuste do tamanho do lote
    validation_data=(test_images, test_labels)
)

# Salvar o modelo treinado
model.save('modelo_treinado.keras')

# Exibir sumário do modelo
model.summary()
