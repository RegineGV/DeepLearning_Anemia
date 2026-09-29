from torchvision import transforms
from src.config import IMAGE_SIZE

# ImageNet statistics for transfer learning
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_train_transforms():
    """
    Data augmentation for training:
    - Resize to target size (224, 224)
    - Random horizontal & vertical flips
    - Slight rotation & color jitter (to handle device and lighting variations)
    - Normalization with ImageNet mean and std
    """
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def get_eval_transforms():
    """
    Deterministic transforms for validation and testing:
    - Resize to target size (224, 224)
    - ToTensor
    - Normalization with ImageNet mean and std
    """
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
