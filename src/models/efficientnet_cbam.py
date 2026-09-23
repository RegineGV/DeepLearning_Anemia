import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights

from src.models.cbam import CBAM

class EfficientNetB0_CBAM(nn.Module):
    """
    Lightweight Deep Learning Model for Non-Invasive Anemia Screening.
    Backbone: EfficientNet-B0 (< 6M total parameters, < 20MB model size)
    Integrated with: CBAM (Convolutional Block Attention Module)
    """
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        reduction_ratio: int = 16,
        kernel_size: int = 7,
    ):
        super().__init__()
        
        # Load EfficientNet-B0 backbone
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base_model = models.efficientnet_b0(weights=weights)

        # Feature extractor: 8 stages, final output channels = 1280
        self.features = base_model.features
        in_features = 1280

        # CBAM Attention Module attached to the bottleneck feature maps
        self.cbam = CBAM(
            in_channels=in_features,
            reduction_ratio=reduction_ratio,
            kernel_size=kernel_size
        )

        # Pooling and Classification Head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract convolutional features: [B, 1280, H/32, W/32]
        x = self.features(x)
        # Apply Channel & Spatial Attention
        x = self.cbam(x)
        # Global Average Pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # Classification logits
        x = self.classifier(x)
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass up to attention-refined feature maps (used for Grad-CAM visualization).
        """
        x = self.features(x)
        x = self.cbam(x)
        return x

    def get_gradcam_target_layer(self) -> nn.Module:
        """
        Returns target layer for Grad-CAM explainability analysis.
        """
        return self.cbam

    def count_parameters(self) -> dict:
        """
        Returns breakdown of total, trainable, and non-trainable parameters.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_params": total,
            "trainable_params": trainable,
            "non_trainable_params": total - trainable,
            "size_mb": total * 4 / (1024 * 1024)  # Size in MB assuming 32-bit float
        }

if __name__ == "__main__":
    model = EfficientNetB0_CBAM(num_classes=2, pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    out = model(dummy_input)
    param_info = model.count_parameters()

    print("=" * 50)
    print("EfficientNet-B0 + CBAM Architecture Summary:")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Total parameters: {param_info['total_params']:,}")
    print(f"Trainable parameters: {param_info['trainable_params']:,}")
    print(f"Model size (FP32): {param_info['size_mb']:.2f} MB")
    print("Under 6M parameters:", param_info['total_params'] < 6_000_000)
    print("Under 20MB size:", param_info['size_mb'] < 20.0)
    print("=" * 50)
