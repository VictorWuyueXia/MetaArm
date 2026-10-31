"""Batched fixed-chart inverse kinematics with exact coupled joint inequalities."""

import hashlib
import time
from pathlib import Path

import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from metaarm.basis import MotionBasis, TaskBasis
from metaarm.pose_model import PoseChart

CHECKPOINT = (
    Path(__file__).resolve().parents[2] / "checkpoints/basis_20261031T105751490589Z_audited.pt"
)
POSITION_TOLERANCE, ANGULAR_TOLERANCE = 1e-3, np.pi / 180


def constrained_step(hessian, gradient, basis, margin):
    """Minimize a positive-definite quadratic over the coupled displacement polytope."""
    step = -torch.linalg.solve(hessian, gradient.unsqueeze(-1)).squeeze(-1)
    a = torch.cat((basis, -basis), dim=1)
    constrained = ((a @ step.unsqueeze(-1)).squeeze(-1) >= margin).any(-1)
    if not constrained.any():
        return step

    # Feasible primal-dual Newton iterates solve only the constrained quadratic rows.
    h, g, a, b = hessian[constrained], gradient[constrained], a[constrained], margin[constrained]
    d, slack, dual = torch.zeros_like(g), b.clone(), torch.ones_like(b)
    if not torch.all(slack > 0):
        raise ValueError("The quadratic step requires a strictly feasible current configuration.")
    for _ in range(24):
        primal = (a @ d.unsqueeze(-1)).squeeze(-1) + slack - b
        stationarity = (
            (h @ d.unsqueeze(-1)).squeeze(-1) + g + (a.mT @ dual.unsqueeze(-1)).squeeze(-1)
        )
        scale = 1 + g.abs().max(-1).values
        converged = (stationarity.abs().max(-1).values <= 1e-8 * scale) & (
            (slack * dual).max(-1).values <= 1e-8 * scale
        )
        if converged.all():
            break
        complement = slack * dual - 0.05 * (slack * dual).mean(-1, keepdim=True)
        matrix = h + a.mT @ ((dual / slack).unsqueeze(-1) * a)
        rhs = -stationarity + (a.mT @ ((complement - dual * primal) / slack).unsqueeze(-1)).squeeze(
            -1
        )
        delta = torch.linalg.solve(matrix, rhs.unsqueeze(-1)).squeeze(-1)
        delta[converged] = 0
        ds = -primal - (a @ delta.unsqueeze(-1)).squeeze(-1)
        dz = (-complement - dual * ds) / slack
        ds[converged], dz[converged] = 0, 0
        ratios = torch.cat((ds / slack, dz / dual), dim=-1)
        fraction = 0.995 / torch.maximum(torch.ones_like(ratios[:, 0]), -ratios.min(-1).values)
        d += fraction[:, None] * delta
        slack += fraction[:, None] * ds
        dual += fraction[:, None] * dz
    if not torch.isfinite(d).all() or not torch.all((a @ d.unsqueeze(-1)).squeeze(-1) < b):
        raise FloatingPointError("The constrained quadratic step is nonfinite or infeasible.")
    step[constrained] = d
    return step


class CoefficientIK:
    """Own independent CUDA kinematics and the explicitly selected trained representation."""

    def __init__(self, model, num_envs, checkpoint):
        if not torch.cuda.is_available() or type(num_envs) is not int or num_envs < 1:
            raise ValueError("Coefficient IK requires CUDA and a positive batch size.")
        wp.config.log_level = wp.LOG_WARNING
        wp.init()
        wp.set_device("cuda:0")
        wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()))
        self.model, self.num_envs = model, num_envs
        joints = np.array([model.joint(f"fr3_joint{i}").id for i in range(1, 8)])
        indices = model.jnt_qposadr[joints]
        self.q_slice, self.site = (
            slice(indices[0], indices[-1] + 1),
            model.site("attachment_site").id,
        )
        self.joints = torch.tensor(joints, device="cuda:0")
        self.bounds = torch.tensor(model.jnt_range[joints], device="cuda:0", dtype=torch.float64)
        self.gpu_model = mjw.put_model(model)
        self.data = mjw.make_data(model, nworld=num_envs, nconmax=1, njmax=1)
        self.qpos = wp.to_torch(self.data.qpos)
        self.qpos.copy_(torch.tensor(model.key("home").qpos, device="cuda:0", dtype=torch.float32))
        self.position = wp.to_torch(self.data.site_xpos)[:, self.site]
        self.rotation = wp.to_torch(self.data.site_xmat)[:, self.site].reshape(num_envs, 3, 3)
        self.anchor, self.axis = wp.to_torch(self.data.xanchor), wp.to_torch(self.data.xaxis)

        # Verify the physical model and load one explicit checkpoint without substitution.
        self.checkpoint = Path(checkpoint).resolve()
        saved = torch.load(self.checkpoint, weights_only=True)
        root = Path(__file__).resolve().parents[2]
        for name, digest in saved["metadata"]["sources"][0]["metadata"]["assets"].items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"The checkpoint physical asset differs: {name}.")
        weights = [
            v
            for k, v in saved["state_dict"].items()
            if k.startswith("layers.") and k.endswith(".weight")
        ]
        outputs = weights[-1].shape[0]
        specification = saved["metadata"]["specification"]["model"]
        if outputs == 84 and len(weights) == 3:
            self.network = MotionBasis(self.bounds, specification).cuda()
        elif outputs == 12:
            self.network = PoseChart(
                self.bounds, specification["width"], specification["side"]
            ).cuda()
        elif outputs in (21, 42):
            task = "position" if outputs == 21 else "pose"
            self.network = TaskBasis(self.bounds, specification, task, len(weights) - 1).cuda()
        else:
            raise ValueError("The checkpoint neural architecture is unsupported.")
        self.network.load_state_dict(saved["state_dict"])
        self.network.eval()
        if not torch.allclose(self.network.joint_limits.double(), self.bounds, atol=1e-6, rtol=0):
            raise ValueError("Checkpoint and robot joint limits differ.")

    def geometry(self, q, differentiate=True):
        """Evaluate tool FK and its base-aligned revolute-joint geometric Jacobian."""
        if q.shape != (self.num_envs, 7) or q.device.type != "cuda":
            raise ValueError("Kinematics requires a matching CUDA configuration batch.")
        self.qpos[:, self.q_slice].copy_(q)
        mjw.kinematics(self.gpu_model, self.data)
        position, rotation = self.position.double(), self.rotation.double()
        if not differentiate:
            return position, rotation, None
        axis, anchor = self.axis[:, self.joints].double(), self.anchor[:, self.joints].double()
        linear = torch.linalg.cross(axis, position[:, None] - anchor, dim=-1).mT
        jacobian = torch.cat((linear, axis.mT), dim=1)
        return position, rotation, jacobian

    def residual(self, q, target, task, differentiate=True):
        """Differentiate position and the principal body rotation logarithm exactly."""
        position, rotation, jacobian = self.geometry(q, differentiate)
        error = (position - target["position"]) / POSITION_TOLERANCE
        derivative = jacobian[:, :3] / POSITION_TOLERANCE if differentiate else None
        if task in ("plane", "position"):
            return error, derivative
        if task != "pose":
            raise ValueError("The IK task must be pose, plane, or position.")

        # The logarithm and inverse right Jacobian use their analytic limits at zero.
        relative = target["rotation"].mT @ rotation
        vee = (
            torch.stack(
                (
                    relative[:, 2, 1] - relative[:, 1, 2],
                    relative[:, 0, 2] - relative[:, 2, 0],
                    relative[:, 1, 0] - relative[:, 0, 1],
                ),
                -1,
            )
            / 2
        )
        angle = torch.atan2(vee.norm(dim=-1), (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2)
        if torch.any(angle >= torch.pi - 1e-4):
            raise ValueError("The pose residual reaches the rotation-logarithm cut.")
        phi = vee / torch.sinc(angle[:, None] / torch.pi)
        if not differentiate:
            return torch.cat((error, phi / ANGULAR_TOLERANCE), -1), None
        skew = torch.zeros_like(relative)
        skew[:, 0, 1], skew[:, 0, 2] = -phi[:, 2], phi[:, 1]
        skew[:, 1, 0], skew[:, 1, 2] = phi[:, 2], -phi[:, 0]
        skew[:, 2, 0], skew[:, 2, 1] = -phi[:, 1], phi[:, 0]
        small = angle < 1e-3
        t = torch.where(small, torch.ones_like(angle), angle)
        coefficient = torch.where(
            small, 1 / 12 + angle.square() / 720, (1 - t / (2 * torch.tan(t / 2))) / t.square()
        )
        inverse_right = (
            torch.eye(3, device="cuda:0") + 0.5 * skew + coefficient[:, None, None] * (skew @ skew)
        )
        angular = inverse_right @ rotation.mT @ jacobian[:, 3:] / ANGULAR_TOLERANCE
        return torch.cat((error, phi / ANGULAR_TOLERANCE), -1), torch.cat((derivative, angular), 1)

    def solve(self, q_ref, basis, target, task, alpha_initial):
        """Optimize only coefficients while preserving the reference and basis throughout."""
        q_ref, basis, alpha = (
            q_ref.detach().double(),
            basis.detach().double(),
            alpha_initial.detach().double().clone(),
        )
        dimension = basis.shape[-1]
        if task not in ("pose", "plane", "position") or dimension not in (
            (6, 7) if task == "pose" else (3, 7)
        ):
            raise ValueError("Expected six pose, three plane, or seven full-joint variables.")
        if (
            q_ref.shape != (self.num_envs, 7)
            or basis.shape != (self.num_envs, 7, dimension)
            or alpha.shape != (self.num_envs, dimension)
        ):
            raise ValueError("The reference, basis, and coefficient batch dimensions differ.")
        if not all(
            torch.isfinite(v).all()
            for v in (q_ref, basis, alpha, target["position"], target["rotation"])
        ):
            raise ValueError("IK data must be finite.")
        if target["position"].shape != (self.num_envs, 3) or target["rotation"].shape != (
            self.num_envs,
            3,
            3,
        ):
            raise ValueError("Targets require matching position and rotation batches.")
        identity = torch.eye(dimension, device="cuda:0", dtype=torch.float64)
        if not torch.allclose(
            basis.mT @ basis, identity.expand(self.num_envs, -1, -1), atol=2e-5, rtol=0
        ):
            raise ValueError("The fixed basis must have orthonormal columns.")
        rotation = target["rotation"].double()
        if (rotation.mT @ rotation - torch.eye(3, device="cuda:0")).abs().max() > 2e-5 or torch.any(
            torch.linalg.det(rotation) <= 0
        ):
            raise ValueError("Target orientations must be proper rotations.")
        q = q_ref + (basis @ alpha.unsqueeze(-1)).squeeze(-1)
        if not torch.all((q > self.bounds[:, 0]) & (q < self.bounds[:, 1])):
            raise ValueError("The coefficient initialization must be strictly joint-feasible.")
        damping = torch.full((self.num_envs,), 1e-3, device="cuda:0", dtype=torch.float64)
        status = torch.full((self.num_envs,), 2, device="cuda:0", dtype=torch.long)
        iterations = torch.zeros_like(status)
        done = torch.zeros(self.num_envs, device="cuda:0", dtype=torch.bool)
        torch.cuda.synchronize()
        started = time.perf_counter()

        # Damped Gauss-Newton uses feasible quadratic steps and explicit cost acceptance.
        for _ in range(100):
            error, derivative = self.residual(q, target, task)
            position_error = error[:, :3].norm(dim=-1) * POSITION_TOLERANCE
            angular_error = error[:, 3:].norm(dim=-1) * ANGULAR_TOLERANCE
            reached = (position_error <= POSITION_TOLERANCE) & (angular_error <= ANGULAR_TOLERANCE)
            status[~done & reached], done = 0, done | reached
            active = ~done
            if not active.any():
                break
            jacobian = derivative @ basis
            gradient = (jacobian.mT @ error.unsqueeze(-1)).squeeze(-1)
            hessian = jacobian.mT @ jacobian
            hessian += damping[:, None, None] * torch.diag_embed(
                hessian.diagonal(dim1=-2, dim2=-1) + 1
            )
            gradient[~active] = 0
            margin = torch.cat((self.bounds[:, 1] - q, q - self.bounds[:, 0]), -1)
            step = constrained_step(hessian, gradient, basis, margin)
            norm = (basis @ step.unsqueeze(-1)).squeeze(-1).norm(dim=-1)
            step *= (0.25 / torch.maximum(norm, torch.full_like(norm, 0.25)))[:, None]
            candidate = q_ref + (basis @ (alpha + step).unsqueeze(-1)).squeeze(-1)
            proposed, _ = self.residual(candidate, target, task)
            if not torch.isfinite(proposed).all():
                raise FloatingPointError("The proposed task residual is nonfinite.")
            accepted = active & (proposed.square().sum(-1) < error.square().sum(-1))
            alpha[accepted] += step[accepted]
            q[accepted] = candidate[accepted]
            damping[active] *= torch.where(accepted[active], 0.3, 10.0)
            iterations[active] += 1
            stalled = active & ((step.norm(dim=-1) < 1e-9) | (damping > 1e12))
            status[stalled], done = 1, done | stalled
        error, _ = self.residual(q, target, task)
        position_error = error[:, :3].norm(dim=-1) * POSITION_TOLERANCE
        angular_error = error[:, 3:].norm(dim=-1) * ANGULAR_TOLERANCE
        margin = torch.minimum(q - self.bounds[:, 0], self.bounds[:, 1] - q).min(-1).values
        success = (
            (position_error <= POSITION_TOLERANCE)
            & (angular_error <= ANGULAR_TOLERANCE)
            & (margin >= 0)
        )
        if not torch.isfinite(q).all() or not torch.isfinite(error).all() or torch.any(margin < 0):
            raise FloatingPointError("IK produced a nonfinite or joint-infeasible candidate.")
        torch.cuda.synchronize()
        return {
            "q": q,
            "alpha": alpha,
            "success": success,
            "status": status,
            "iterations": iterations,
            "position_error": position_error,
            "angular_error": angular_error,
            "joint_margin": margin,
            "seconds": time.perf_counter() - started,
        }
