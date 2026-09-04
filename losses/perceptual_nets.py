"""VGG feature extractors used by the perceptual loss.

Extracted verbatim from ``CNN_deblurring_model_pytorch.py`` (lines 595-716 of the
predecessor repo), which was a 52 KB module reached only by a star-import. The
paper's confs use ``perceptualNet: vgg.vgg16`` or ``vgg.vgg19_bn``, so only the
vgg pair is needed; the resnet/densenet/inception variants alongside them were
referenced by no conf and are not carried over.

``PerceptualLoss.config_perceptual_net`` resolves the class by name
(``vggLossNetwork``), so the class name must not change.

The predecessor also had a ``WeightedVGGLossNetwork`` with learnable per-layer
weights, selected by ``optimize_perceptual``. That is a separate line of work,
not part of this paper, and is not carried over.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class vggLossNetwork(torch.nn.Module):
    def __init__(self, vgg_model, layers=None, data_range=559.0):  # ADD data_range
        super(vggLossNetwork, self).__init__()
        if layers is None:
            layers = ['4', '9']  # relu1_2 and relu2_2
        self.vgg_layers = vgg_model.features
        self.layers_to_use = layers
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
        self.data_range = data_range  # Store your actual data range (559)

    def forward(self, x):
        out = []

        # Store original if needed
        if '-1' in self.layers_to_use:
            out.append(x)

        # CRITICAL FIX: Normalize to [0, 1] first!
        #x = x / self.data_range  # Divide by 559 to get [0, 1]

        # Handle grayscale -> RGB
        if x.shape[1] == 1:
            x = torch.cat((x, x, x), dim=1)

        # Apply ImageNet normalization
        # Move mean and std to device and reshape for broadcasting
        self.mean = self.mean.view(1, 3, 1, 1).to(x.device)
        self.std = self.std.view(1, 3, 1, 1).to(x.device)
        #x = (x - self.mean) / self.std

        # Forward through VGG layers
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layers_to_use:
                out.append(x)

        return out
