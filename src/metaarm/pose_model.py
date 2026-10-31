"""Target-conditioned six-coordinate pose charts with a learned feasible seed."""

import torch
from torch import nn


def pose_geometry(jacobian):
    """Construct an exact right inverse and a continuous one-dimensional null gauge."""
    jacobian = jacobian.double()
    gram = jacobian @ jacobian.mT
    inverse = torch.cholesky_solve(jacobian, torch.linalg.cholesky(gram)).mT
    null = torch.eye(7, device=jacobian.device, dtype=torch.float64)[2] - (
        inverse @ jacobian[:, :, 2:3]
    ).squeeze(-1)
    length = null.norm(dim=-1, keepdim=True)
    if torch.any(length < 1e-6):
        raise ValueError("The declared third-joint null gauge vanishes at this reference.")
    return inverse, null / length


class PoseChart(nn.Module):
    """Map a reference and requested pose displacement to one affine search chart."""

    def __init__(self, limits, width, side):
        super().__init__()
        self.register_buffer("joint_limits", torch.as_tensor(limits, dtype=torch.float32))
        self.register_buffer("scales", torch.tensor([side / 2] * 3 + [torch.pi / 18] * 3))
        self.layers = nn.Sequential(
            nn.Linear(13, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 12),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, q_ref, geometry, displacement):
        """Preserve reference task rank and initialize strictly inside the joint polytope."""
        lower, upper = self.joint_limits.T
        inputs = torch.cat(
            (2 * (q_ref - lower) / (upper - lower) - 1, displacement.float() / self.scales), -1
        )
        output = self.layers(inputs.float()).double()
        inverse, null = geometry
        graph = 4 * torch.tanh(output[:, :6])
        frame = inverse + null[:, :, None] * graph[:, None, :]
        factor = torch.linalg.cholesky(frame.mT @ frame)
        basis = torch.linalg.solve_triangular(factor, frame.mT, upper=False).mT
        tangent = displacement.double() + 2 * self.scales * torch.tanh(output[:, 6:])
        delta = (frame @ tangent[:, :, None]).squeeze(-1)
        # The ray fraction constructs a feasible initialization without changing the chart.
        margin = torch.where(delta >= 0, upper - q_ref, q_ref - lower).double()
        fraction = torch.minimum(
            torch.ones_like(delta[:, 0]),
            (0.995 * margin / delta.abs().clamp_min(1e-30)).min(-1).values,
        )
        delta = delta * fraction[:, None]
        alpha = (basis.mT @ delta[:, :, None]).squeeze(-1)
        return basis, alpha, graph

    def loss(self, q_ref, geometry, displacement, label):
        """Fit joint seeds and representation error on independently feasible teacher labels."""
        basis, alpha, graph = self(q_ref, geometry, displacement)
        delta = (basis @ alpha[:, :, None]).squeeze(-1)
        projected = (basis @ (basis.mT @ label[:, :, None])).squeeze(-1)
        seed_mse = (delta - label).square().mean()
        objective = seed_mse + 4 * (projected - label).square().mean()
        return objective + 1e-5 * graph.square().mean(), seed_mse
