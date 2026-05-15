# TCC2 — Aumento de Dados com Albumentations para Classificação de Imagens

Trabalho de Conclusão de Curso que investiga o impacto de técnicas de data augmentation na performance de uma Rede Neural Convolucional (CNN) treinada no dataset CIFAR-10.

## Sobre o Projeto

O projeto implementa um pipeline de aumento de dados utilizando a biblioteca Albumentations e avalia como diferentes transformações afetam a acurácia de um modelo CNN construído com TensorFlow/Keras.

### Estrutura

- `augmentation_pipeline.py` — Define o pipeline de transformações de aumento de dados com Albumentations.
- Treinamento e avaliação do modelo CNN com arquitetura Sequential (Conv2D, MaxPooling, BatchNormalization, Dropout).
- Visualização dos resultados com Matplotlib.

## Tecnologias

- Python 3.12
- TensorFlow / Keras
- Albumentations
- OpenCV
- Matplotlib
- NumPy

## Como Executar

1. Clone o repositório:

```bash
git clone https://github.com/Richoland/tcc2.git
cd tcc2
```

2. Crie e ative um ambiente virtual:

```bash
python -m venv venv

# Windows
.\venv\Scripts\Activate

# Linux/Mac
source venv/bin/activate
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

4. Execute o projeto:

```bash
python main.py
```

## Dataset

O projeto utiliza o [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), carregado diretamente via `tensorflow.keras.datasets`. O dataset contém 60.000 imagens 32x32 em 10 classes.