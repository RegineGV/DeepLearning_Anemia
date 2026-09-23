import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

from src.config import FIGURES_DIR
from src.data.transforms import get_eval_transforms

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for Explainable AI (XAI).
    Visualizes which visual regions triggered the anemia prediction.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module = None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer or model.get_gradcam_target_layer()

        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Generates normalized Grad-CAM heatmap [0, 1] for the specified class index.
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        # Backward target score
        score = output[0, class_idx]
        score.backward()

        # Global average pooling of gradients: [C]
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        # Normalize between 0 and 1
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def overlay_heatmap(
        self,
        pil_image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap_name: str = "jet"
    ) -> Image.Image:
        """
        Overlays heatmap on top of original PIL image using matplotlib and PIL.
        """
        w, h = pil_image.size
        
        # Resize heatmap using PIL bilinear interpolation
        heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize((w, h), Image.Resampling.BILINEAR)
        heatmap_norm = np.array(heatmap_img) / 255.0

        # Apply colormap
        cmap = plt.get_cmap(colormap_name)
        colored_heatmap = (cmap(heatmap_norm)[:, :, :3] * 255).astype(np.uint8)

        # Overlay blend with original image
        orig_np = np.array(pil_image.convert("RGB"))
        blended = np.uint8(alpha * colored_heatmap + (1 - alpha) * orig_np)
        return Image.fromarray(blended)

