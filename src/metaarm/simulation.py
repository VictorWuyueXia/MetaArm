"""Batched GPU dynamics with named joint views shared by Warp and PyTorch."""

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp


class Simulation:
    """Own the FR3 worlds and keep all stepping on one shared CUDA stream."""

    def __init__(self, model: mujoco.MjModel, num_envs: int):
        # Bind both libraries to the same explicit device and execution stream.
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be a positive integer.")
        if not torch.cuda.is_available():
            raise RuntimeError("MetaArm requires CUDA PyTorch and an NVIDIA GPU.")
        wp.config.log_level = wp.LOG_WARNING
        wp.init()
        wp.set_device("cuda:0")
        self.torch_stream = torch.cuda.Stream(device=0)
        torch.cuda.set_stream(self.torch_stream)
        self.stream = wp.stream_from_torch(self.torch_stream)
        wp.set_stream(self.stream)
        self.model, self.num_envs = model, num_envs
        self.kinematics = mujoco.MjData(model)
        joints = np.array([model.joint(f"fr3_joint{i}").id for i in range(1, 8)])
        self.q_indices = model.jnt_qposadr[joints].copy()
        self.v_indices = model.jnt_dofadr[joints].copy()
        actuators = np.array([model.actuator(f"fr3_joint{i}").id for i in range(1, 8)])
        self.q_slice = slice(self.q_indices[0], self.q_indices[-1] + 1)
        self.v_slice = slice(self.v_indices[0], self.v_indices[-1] + 1)
        self.u_slice = slice(actuators[0], actuators[-1] + 1)
        self.site = model.site("attachment_site").id
        self.bounds = torch.tensor(model.jnt_range[joints], device="cuda:0", dtype=torch.float32)
        self.home = torch.tensor(
            model.key("home").qpos[self.q_indices], device="cuda:0", dtype=torch.float32
        )

        # Allocate once and retain array views rather than copying state each step.
        self.gpu_model = mjw.put_model(model)
        self.data = mjw.make_data(model, nworld=num_envs, nconmax=64, njmax=256)
        self.qpos, self.qvel = wp.to_torch(self.data.qpos), wp.to_torch(self.data.qvel)
        self.ctrl = wp.to_torch(self.data.ctrl)
        self.q, self.qd = self.qpos[:, self.q_slice], self.qvel[:, self.v_slice]
        self.xpos, self.xmat = wp.to_torch(self.data.xpos), wp.to_torch(self.data.xmat)
        self.site_xpos = wp.to_torch(self.data.site_xpos)
        self.site_xmat = wp.to_torch(self.data.site_xmat)
        self.time = wp.to_torch(self.data.time)
        self.reset(self.home)
        # Capture the warmed control interval to remove Python kernel-launch overhead.
        for _ in range(5):
            mjw.step(self.gpu_model, self.data)
        mjw.forward(self.gpu_model, self.data)
        wp.synchronize_stream(self.stream)
        self.reset(self.home)
        wp.capture_begin(stream=self.stream)
        for _ in range(5):
            mjw.step(self.gpu_model, self.data)
        mjw.forward(self.gpu_model, self.data)
        self.graph = wp.capture_end(stream=self.stream)
        self.reset(self.home)
        wp.synchronize_stream(self.stream)

    def reset(self, q_initial: torch.Tensor) -> None:
        """Reset full dynamic state and apply the explicit arm initial condition."""
        # Reject invalid configurations before changing any world.
        if q_initial.shape not in ((7,), (self.num_envs, 7)):
            raise ValueError("Reset configuration must have shape (7,) or (num_envs, 7).")
        if (
            q_initial.device != self.q.device
            or q_initial.dtype != torch.float32
            or not torch.isfinite(q_initial).all()
        ):
            raise ValueError("Reset configuration must be finite and on cuda:0.")
        if not torch.all((q_initial >= self.bounds[:, 0]) & (q_initial <= self.bounds[:, 1])):
            raise ValueError("Reset configuration violates an FR3 joint limit.")
        mjw.reset_data(self.gpu_model, self.data)
        self.q.copy_(q_initial)
        self.ctrl[:, self.u_slice].copy_(q_initial)
        mjw.forward(self.gpu_model, self.data)

    def step(self, q_ref: torch.Tensor) -> None:
        """Advance five physics steps, then refresh geometry at the integrated state."""
        # Position commands are physical actuator references, never state assignments.
        if (
            q_ref.shape != (self.num_envs, 7)
            or q_ref.device != self.q.device
            or q_ref.dtype != torch.float32
        ):
            raise ValueError("Joint reference must have shape (num_envs, 7) on cuda:0.")
        if not torch.isfinite(q_ref).all() or not torch.all(
            (q_ref >= self.bounds[:, 0]) & (q_ref <= self.bounds[:, 1])
        ):
            raise ValueError("Joint reference must be finite and inside the FR3 limits.")
        self.ctrl[:, self.u_slice].copy_(q_ref)
        wp.capture_launch(self.graph, stream=self.stream)
        if not torch.isfinite(self.qpos).all() or not torch.isfinite(self.qvel).all():
            raise FloatingPointError("The simulated state became non-finite.")
