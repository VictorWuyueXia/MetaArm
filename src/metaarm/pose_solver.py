"""Gain-ratio pose search with exact coupled joint inequalities and fixed charts."""

import time

import torch

from metaarm.gpu_ik import ANGULAR_TOLERANCE, POSITION_TOLERANCE, constrained_step


@torch.no_grad()
def solve_pose(solver, q_ref, basis, target, alpha_initial):
    """Solve six-dimensional pose residuals with a shared reduced/full-joint LM policy."""
    if q_ref.ndim not in (2, 3) or (q_ref.ndim == 3 and q_ref.shape[1] != 3):
        raise ValueError("Pose search accepts one chart or an explicit three-chart axis.")
    branches = q_ref.shape[1] if q_ref.ndim == 3 else 1
    q_ref = q_ref.reshape(-1, 7).double()
    dimension = basis.shape[-1]
    basis = basis.reshape(-1, 7, dimension).double()
    alpha = alpha_initial.reshape(-1, dimension).double().clone()
    target = {
        "position": target["position"].reshape(-1, 3),
        "rotation": target["rotation"].reshape(-1, 3, 3),
    }
    n, _, dimension = basis.shape
    if n != solver.num_envs or dimension not in (6, 7):
        raise ValueError("Pose search requires six or seven coordinates in the declared batch.")
    if not torch.allclose(
        basis.mT @ basis,
        torch.eye(dimension, device="cuda:0", dtype=torch.float64).expand(n, -1, -1),
        atol=2e-5,
        rtol=0,
    ):
        raise ValueError("The fixed search chart must be orthonormal.")
    if not all(
        torch.isfinite(value).all()
        for value in (q_ref, basis, alpha, target["position"], target["rotation"])
    ):
        raise ValueError("Pose search inputs must be finite.")
    rotation = target["rotation"].double()
    if (rotation.mT @ rotation - torch.eye(3, device="cuda:0")).abs().max() > 2e-5 or torch.any(
        torch.linalg.det(rotation) <= 0
    ):
        raise ValueError("Pose targets must contain proper rotation matrices.")
    q = q_ref + (basis @ alpha[:, :, None]).squeeze(-1)
    if not torch.all((q > solver.bounds[:, 0]) & (q < solver.bounds[:, 1])):
        raise ValueError("Pose initialization must be strictly joint-feasible.")
    damping = torch.full((n,), 1e-3, device="cuda:0", dtype=torch.float64)
    multiplier = torch.full_like(damping, 2)
    status = torch.full((n,), 2, device="cuda:0", dtype=torch.long)
    iterations = torch.zeros_like(status)
    done = torch.zeros(n, device="cuda:0", dtype=torch.bool)
    torch.cuda.synchronize()
    started = time.perf_counter()
    # Gain-ratio damping adapts to the local model rather than a fixed accept/reject schedule.
    for _ in range(100):
        error, derivative = solver.residual(q, target, "pose")
        reached = (error[:, :3].norm(dim=-1) <= 1) & (error[:, 3:].norm(dim=-1) <= 1)
        completed = reached.reshape(-1, branches).any(-1).repeat_interleave(branches)
        status[~done & reached] = 0
        status[~done & completed & ~reached] = 3
        done = done | completed
        active = ~done
        if not active.any():
            break
        jacobian = derivative @ basis
        gradient = (jacobian.mT @ error[:, :, None]).squeeze(-1)
        normal = jacobian.mT @ jacobian
        hessian = normal + damping[:, None, None] * torch.diag_embed(
            normal.diagonal(dim1=-2, dim2=-1) + 1
        )
        gradient[~active] = 0
        margin = torch.cat((solver.bounds[:, 1] - q, q - solver.bounds[:, 0]), -1)
        step = constrained_step(hessian, gradient, basis, margin)
        norm = (basis @ step[:, :, None]).squeeze(-1).norm(dim=-1)
        step *= (0.25 / norm.clamp_min(0.25))[:, None]
        candidate = q_ref + (basis @ (alpha + step)[:, :, None]).squeeze(-1)
        proposed, _ = solver.residual(candidate, target, "pose", differentiate=False)
        prediction = -(gradient * step).sum(-1) - 0.5 * (
            step * (normal @ step[:, :, None]).squeeze(-1)
        ).sum(-1)
        reduction = 0.5 * (error.square().sum(-1) - proposed.square().sum(-1))
        ratio = reduction / prediction.clamp_min(1e-30)
        accepted = active & (reduction > 0) & (prediction > 0)
        alpha[accepted] += step[accepted]
        q[accepted] = candidate[accepted]
        damping[accepted] *= torch.maximum(
            torch.full_like(ratio[accepted], 1 / 3), 1 - (2 * ratio[accepted] - 1).pow(3)
        )
        multiplier[accepted] = 2
        rejected = active & ~accepted
        damping[rejected] *= multiplier[rejected]
        multiplier[rejected] *= 2
        iterations[active] += 1
        stalled = active & ((step.norm(dim=-1) < 1e-9) | (damping > 1e12))
        status[stalled], done = 1, done | stalled
    error, _ = solver.residual(q, target, "pose", differentiate=False)
    position_error = error[:, :3].norm(dim=-1) * POSITION_TOLERANCE
    angular_error = error[:, 3:].norm(dim=-1) * ANGULAR_TOLERANCE
    margin = torch.minimum(q - solver.bounds[:, 0], solver.bounds[:, 1] - q).min(-1).values
    if not torch.isfinite(error).all() or torch.any(margin < 0):
        raise FloatingPointError("Pose search produced a nonfinite or infeasible candidate.")
    success = (position_error <= POSITION_TOLERANCE) & (angular_error <= ANGULAR_TOLERANCE)
    status[success] = 0
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
