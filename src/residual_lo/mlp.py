import math

import torch
import torch.nn as nn


class MLP(nn.Module):
    @staticmethod
    def from_parameter_budget(
        in_features: int,
        out_features: int,
        parameter_budget: int,
    ) -> "MLP":
        """
        Create a two-hidden-layer MLP as close as possible to a parameter budget.

        Both hidden layers use the same width. Since the parameter count is
        monotonic in that width, the quadratic parameter-count formula gives
        the nearest candidate widths directly.
        """
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if parameter_budget <= 0:
            raise ValueError("parameter_budget must be positive")

        def parameter_count(width: int) -> int:
            return (
                (in_features + 1) * width
                + (width + 1) * width
                + (width + 1) * out_features
            )

        linear_coefficient = in_features + out_features + 2
        approximate_width = (
            -linear_coefficient
            + math.sqrt(linear_coefficient**2 + 4 * (parameter_budget - out_features))
        ) / 2
        candidates = {
            max(1, math.floor(approximate_width)),
            max(1, math.ceil(approximate_width)),
        }
        width = min(
            candidates,
            key=lambda candidate: (
                abs(parameter_count(candidate) - parameter_budget),
                parameter_count(candidate),
            ),
        )
        return MLP(
            in_features=in_features,
            out_features=out_features,
            hidden_sizes=[width, width],
        )

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_sizes: list[int] | None = None,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_sizes = hidden_sizes
        if hidden_sizes is None:
            hidden_sizes = [128, 128]
        self.hidden_sizes = hidden_sizes
        self.layers = nn.ModuleList([nn.Linear(in_features, hidden_sizes[0])])
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
        self.layers.append(nn.Linear(hidden_sizes[-1], out_features))
        self.activation = activation

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)
