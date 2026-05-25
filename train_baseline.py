from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Carregar CIFAR-10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Sem augmentation: usar os dados originais
train_images = train_images / 255.0
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
    train_images, train_labels,
    epochs=25,  # Aumente o número de épocas para melhor convergência
    batch_size=64,  # Ajuste do tamanho do lote
    validation_data=(test_images, test_labels)
)

# Salvar o modelo treinado versão base
model.save('modelo_baseline.keras')

# Exibir sumário do modelo
model.summary()
