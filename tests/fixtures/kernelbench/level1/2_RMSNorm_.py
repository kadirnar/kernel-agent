import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Tiny test problem in KernelBench format: RMS Normalization over the feature dimension.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super(Model, self).__init__()  # noqa: UP008 (as in KernelBench)
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        rms = torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.eps)
        return x / rms


batch_size = 4
features = 16
dim1 = 8
dim2 = 8


def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]


def get_init_inputs():
    return [features]
