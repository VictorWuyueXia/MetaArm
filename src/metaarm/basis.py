"""Reference geometry and neural nullspace corrections for joint-motion subspaces."""

import torch
from torch import nn


def reference_geometry(J_ref, task):
    """Compute the fixed task QR and right inverse once for a reference batch."""
    if task not in ("pose", "plane", "position") or J_ref.shape[-2:] != (6, 7):
        raise ValueError("Expected pose/plane/position and a (..., 6, 7) reference Jacobian.")
    dimension = 6 if task == "pose" else 3
    jacobian = J_ref[..., :dimension, :]
    if not torch.isfinite(jacobian).all() or torch.any(
        torch.linalg.svdvals(jacobian)[..., -1] <= 1e-6
    ):
        raise ValueError("The reference task Jacobian is numerically rank deficient.")
    v, h = torch.linalg.qr(jacobian.mT, mode="reduced")
    return v, torch.linalg.solve_triangular(h, v.mT, upper=True).mT


class MotionBasis(nn.Module):
    """Learn a bounded nullspace graph with optional geometric input symmetries."""

    def __init__(self, joint_limits, specification):
        super().__init__()
        self.specification = dict(specification)
        if set(specification) != {"width", "scale", "normal_features", "equivariant"}:
            raise ValueError("The model specification has exactly four experiment fields.")
        width = specification["width"]
        if type(width) is not int or width < 1 or specification["scale"] <= 0:
            raise ValueError("Width and correction scale must be positive.")
        limits = torch.as_tensor(joint_limits, dtype=torch.float32).detach().clone()
        if (
            limits.shape != (7, 2)
            or not torch.isfinite(limits).all()
            or not torch.all(limits[:, 1] > limits[:, 0])
        ):
            raise ValueError("The network requires seven finite ordered joint-limit pairs.")
        self.register_buffer("joint_limits", limits)
        inputs = 13 if specification["normal_features"] else 7
        self.layers = nn.Sequential(
            nn.Linear(inputs, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 84),
            nn.Tanh(),
        )

    def forward(self, q_ref, geometry, task, normal, orthonormal=True):
        """Return the QR basis, or its raw graph for the equivalent projector loss."""
        if task not in ("pose", "plane"):
            raise ValueError("The basis task must be pose or plane.")
        dimension = 6 if task == "pose" else 3
        v, right_inverse = geometry
        if q_ref.shape[-1:] != (7,) or v.shape != (*q_ref.shape[:-1], 7, dimension):
            raise ValueError("Reference configurations and cached task geometry do not match.")
        lower, upper = self.joint_limits.unbind(-1)
        if not torch.isfinite(q_ref).all() or not torch.all((q_ref > lower) & (q_ref < upper)):
            raise ValueError("Every basis reference must be finite and interior to its limits.")
        if task == "pose" and normal is not None:
            raise ValueError("Pose bases do not take a normal.")
        if task == "plane" and (normal is None or normal.shape != (*q_ref.shape[:-1], 3)):
            raise ValueError("Plane bases require matching (..., 3) normals.")
        if task == "plane" and (
            not torch.isfinite(normal).all()
            or not torch.allclose(
                normal.norm(dim=-1), torch.ones_like(normal[..., 0]), atol=1e-5, rtol=0
            )
        ):
            raise ValueError("Plane normals must be finite unit vectors.")

        # Base-yaw canonicalization removes a known rotation from the learned relation.
        x = 2 * (q_ref - lower) / (upper - lower) - 1
        if self.specification["equivariant"]:
            c, s = q_ref[..., 0].cos(), q_ref[..., 0].sin()
            rotation = torch.zeros((*q_ref.shape[:-1], 3, 3), device=x.device, dtype=x.dtype)
            rotation[..., 0, 0], rotation[..., 0, 1] = c, -s
            rotation[..., 1, 0], rotation[..., 1, 1], rotation[..., 2, 2] = s, c, 1
            x = torch.cat((torch.zeros_like(x[..., :1]), x[..., 1:]), dim=-1)
            if task == "plane":
                normal = (rotation.mT @ normal.unsqueeze(-1)).squeeze(-1)
        projector = None if task == "pose" else normal.unsqueeze(-1) * normal.unsqueeze(-2)
        if self.specification["normal_features"]:
            features = torch.zeros((*x.shape[:-1], 6), device=x.device, dtype=x.dtype)
            if task == "plane":
                features = projector[..., [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]
            x = torch.cat((x, features), dim=-1)

        # Symmetric normal features and projectors preserve the n/-n plane identity.
        neural = self.layers(x)
        if task == "pose":
            u = neural[..., :42].reshape(*q_ref.shape[:-1], 7, 6)
        else:
            tangent = neural[..., 42:63].reshape(*q_ref.shape[:-1], 7, 3)
            axial = neural[..., 63:].reshape(*q_ref.shape[:-1], 7, 3)
            u = tangent + (axial - tangent) @ projector
        if self.specification["equivariant"]:
            u = (
                u @ rotation.mT
                if task == "plane"
                else torch.cat((u[..., :3] @ rotation.mT, u[..., 3:] @ rotation.mT), dim=-1)
            )
        raw = right_inverse + self.specification["scale"] * (u - v @ (v.mT @ u))
        return torch.linalg.qr(raw, mode="reduced")[0] if orthonormal else raw


class TaskBasis(nn.Module):
    """Learn a bounded regional position or pose chart with an explicit architecture."""

    def __init__(self, joint_limits, specification, task, depth):
        super().__init__()
        if task not in ("position", "pose") or depth not in (2, 3):
            raise ValueError("Regional models require position/pose and two/three hidden layers.")
        self.task, self.dimension = task, 3 if task == "position" else 6
        if set(specification) != {"width", "scale"}:
            raise ValueError("Regional models require only width and scale.")
        width = specification["width"]
        if type(width) is not int or width < 1 or specification["scale"] <= 0:
            raise ValueError("Width and nullspace scale must be positive.")
        self.specification = dict(specification)
        limits = torch.as_tensor(joint_limits, dtype=torch.float32).detach().clone()
        if (
            limits.shape != (7, 2)
            or not torch.isfinite(limits).all()
            or torch.any(limits[:, 0] >= limits[:, 1])
        ):
            raise ValueError("Regional models require seven finite ordered joint limits.")
        self.register_buffer("joint_limits", limits)
        layers = [nn.Linear(7, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend((nn.Linear(width, width), nn.Tanh()))
        layers.extend((nn.Linear(width, 7 * self.dimension), nn.Tanh()))
        self.layers = nn.Sequential(*layers)

    def forward(self, q_ref, geometry, orthonormal=True):
        """Construct one fixed task-sized chart at each explicit reference."""
        lower, upper = self.joint_limits.unbind(-1)
        v, right_inverse = geometry
        if q_ref.shape[-1] != 7 or v.shape != (*q_ref.shape[:-1], 7, self.dimension):
            raise ValueError("Regional references and cached geometry must have matching shapes.")
        if not torch.isfinite(q_ref).all() or not torch.all((q_ref > lower) & (q_ref < upper)):
            raise ValueError("Chart references must be finite and strictly joint-interior.")
        x = 2 * (q_ref - lower) / (upper - lower) - 1
        neural = self.layers(x).reshape(*q_ref.shape[:-1], 7, self.dimension)
        raw = right_inverse + self.specification["scale"] * (neural - v @ (v.mT @ neural))
        return torch.linalg.qr(raw, mode="reduced")[0] if orthonormal else raw

    def reconstruction(self, q_ref, geometry, covariance):
        """Compute exact uncentered all-waypoint displacement MSE by a small projector."""
        raw = self(q_ref, geometry, False).double()
        factor, info = torch.linalg.cholesky_ex(raw.mT @ raw)
        if torch.any(info != 0):
            raise FloatingPointError("The position-chart Gram matrix is not positive definite.")
        residual = torch.eye(7, device=raw.device, dtype=raw.dtype) - raw @ torch.cholesky_solve(
            raw.mT, factor
        )
        error = (residual @ covariance @ residual.mT).diagonal(dim1=-2, dim2=-1).sum(-1) / 7
        loss = error.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("The position reconstruction loss is nonfinite.")
        return loss
