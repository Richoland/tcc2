import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 1. Carregar o modelo treinado
model = load_model('modelo_treinado.keras')  # Substitua pelo caminho correto

# 2. Carregar os dados de teste
(_, _), (test_images, test_labels) = cifar10.load_data()

# Pré-processar os dados de teste (normalização)
test_images = test_images / 255.0

# 3. Avaliar o modelo no conjunto de teste
loss, accuracy = model.evaluate(test_images, test_labels, verbose=2)
print(f"Perda no teste: {loss:.4f}")
print(f"Precisão no teste: {accuracy:.4f}")

# 4. Fazer previsões
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)

# 5. Métricas avançadas (Precision, Recall, F1-score)
print("\nRelatório de Classificação:")
print(classification_report(test_labels, predicted_classes))

# 6. Matriz de Confusão
conf_matrix = confusion_matrix(test_labels, predicted_classes)

# Exibir a Matriz de Confusão
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.title("Matriz de Confusão")
plt.xlabel("Classe Predita")
plt.ylabel("Classe Verdadeira")
plt.show()

# 7. Mostrar exemplos de previsões
def plot_image_and_prediction(index):
    """
    Mostra uma imagem do conjunto de teste com a previsão e o valor verdadeiro.
    """
    plt.imshow(test_images[index])
    plt.title(f"Verdade: {test_labels[index][0]}, Predição: {predicted_classes[index]}")
    plt.axis('off')
    plt.show()

# Mostrar 5 exemplos aleatórios
for i in range(5):
    plot_image_and_prediction(i)
