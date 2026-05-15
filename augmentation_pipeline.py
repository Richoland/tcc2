import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_augmentation_pipeline(for_training=False):
    """
    Retorna um pipeline de aumentação configurável.

    Args:
        for_training (bool): Se True, adiciona a conversão para tensor (necessário para treinamento).
    """
    pipeline = [
        A.Rotate(limit=30, p=0.5),
        A.Flip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(p=0.5),
    ]
    if for_training:
        pipeline.append(ToTensorV2())

    return A.Compose(pipeline)
