import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Tiny test problem in KernelBench format: a linear layer followed by a scale.
    """

    def __init__(self, in_features: int, out_features: int, scale: float):
        super(Model, self).__init__()  # noqa: UP008 (as in KernelBench)
        self.linear = nn.Linear(in_features, out_features)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        return self.linear(x) * self.scale


batch_size = 512
in_features = 64
out_features = 32
scale = 0.5


def get_inputs():
    x = torch.rand(batch_size, in_features)
    return [x]


def get_init_inputs():
    return [in_features, out_features, scale]
