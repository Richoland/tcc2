from augmentation_pipeline import get_augmentation_pipeline
import cv2
import matplotlib.pyplot as plt

# Carregar uma imagem
image_path = 'tenis2.png'  # Substitua pelo caminho de uma imagem
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Obter pipeline para validação
augmentation_pipeline = get_augmentation_pipeline(for_training=False)

# Aplicar o pipeline
augmented = augmentation_pipeline(image=image)

# Mostrar imagem transformada
plt.imshow(augmented['image'])
plt.title('Transformação Teste')
plt.axis('off')
plt.show()
