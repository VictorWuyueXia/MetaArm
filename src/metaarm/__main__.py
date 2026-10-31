"""Launch GPU dynamics with explicit learned/full-joint inverse-kinematics comparisons."""

import time
import tomllib
from pathlib import Path
from queue import SimpleQueue

import numpy as np
import torch
import viser
import viser.transforms as vtf
from mjviser import ViserMujocoScene

from metaarm.basis import reference_geometry
from metaarm.gpu_ik import CHECKPOINT, CoefficientIK
from metaarm.scene import build_scene
from metaarm.simulation import Simulation


def main() -> None:
    """Own the simulation, fixed-chart comparisons, and queued browser experiment actions."""
    root = Path(__file__).resolve().parents[2]
    experiment = tomllib.loads((root / "experiments/reach.toml").read_text())
    if set(experiment) != {"num_envs", "target_poses", "motion_seconds"}:
        raise ValueError("The reaching experiment has exactly three configuration fields.")
    if not np.isfinite(experiment["motion_seconds"]) or experiment["motion_seconds"] <= 0:
        raise ValueError("motion_seconds must be finite and positive.")
    targets = np.asarray(experiment["target_poses"], dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != 7 or len(targets) == 0:
        raise ValueError("target_poses must contain xyz and unit wxyz poses.")
    print("Loading FR3 CUDA physics and trained coefficient IK.", flush=True)
    sim = Simulation(build_scene(), experiment["num_envs"])
    solver = CoefficientIK(sim.model, 1, CHECKPOINT)

    # Controls declare task geometry and the successful comparison result to drive.
    server = viser.ViserServer(host="127.0.0.1", port=8080, label="MetaArm · Franka FR3")
    if server.get_port() != 8080:
        server.stop()
        raise RuntimeError("The fixed UI port 8080 is already occupied.")
    server.initial_camera.position, server.initial_camera.look_at = (
        (1.6, -1.5, 1.2),
        (0.35, 0.0, 0.35),
    )
    scene_view = ViserMujocoScene(server, sim.model, num_envs=1)
    server.gui.add_markdown(
        "# MetaArm\nFR3 · GPU MuJoCo\n\nApply compares both IK methods at the current reference."
    )
    task = server.gui.add_dropdown("Task", ("Pose", "Plane"))
    method = server.gui.add_dropdown("Drive with", ("Learned coefficients", "Full joints"))
    presets = server.gui.add_dropdown(
        "Preset", tuple(f"Target {i + 1}" for i in range(len(targets)))
    )
    position = server.gui.add_vector3("Requested xyz (m)", targets[0, :3], step=0.01)
    orientation = server.gui.add_vector3(
        "Target rpy (deg)", np.rad2deg(vtf.SO3(targets[0, 3:]).as_rpy_radians()), step=5.0
    )
    direction = server.gui.add_vector3(
        "Plane direction",
        (0.0, 0.0, 1.0),
        step=0.1,
        disabled=True,
        hint="Nonzero direction; normalized to the plane normal.",
    )
    server.gui.add_markdown(
        "Plane mode projects requested xyz onto the plane through the current tool. "
        "Orientation is free."
    )
    marker = server.scene.add_frame(
        "/target", position=targets[0, :3], wxyz=targets[0, 3:], axes_length=0.12, axes_radius=0.004
    )
    plane = server.scene.add_grid(
        "/task-plane", width=0.5, height=0.5, cell_size=0.05, section_size=0.25, visible=False
    )
    comparison = server.gui.add_markdown(
        "Apply a target to compare six/three learned coefficients with seven joints."
    )
    telemetry = server.gui.add_markdown("Initializing experiment…")
    requests = SimpleQueue()

    # Callbacks capture complete requests; only the owning loop changes physical state.
    def enqueue(event) -> None:
        if event.client is not None:
            requests.put(
                (
                    event.target.label,
                    event.target.value,
                    position.value,
                    orientation.value,
                    task.value,
                    method.value,
                    direction.value,
                )
            )

    presets.on_update(enqueue)
    task.on_update(enqueue)
    for action in ("Run / Pause", "Reset", "Apply target", "Quit"):
        server.gui.add_button(action).on_click(enqueue)
    q_start, q_goal = sim.q.clone(), sim.home.expand_as(sim.q)
    active_pose, active_task = targets[0].copy(), "pose"
    origin, normal = np.zeros(3), np.array([0.0, 0.0, 1.0])
    running, exiting, motion_step, next_render = False, False, 0, 0.0
    print(
        "MetaArm ready: http://127.0.0.1:8080 — GPU coefficient/full IK; starts paused.", flush=True
    )

    while not exiting:
        tick = time.perf_counter()
        while not requests.empty():
            action, value, xyz, rpy, task_name, method_name, plane_direction = requests.get()
            if action == "Preset":
                selected = targets[int(value.split()[-1]) - 1]
                position.value = tuple(selected[:3])
                orientation.value = tuple(np.rad2deg(vtf.SO3(selected[3:]).as_rpy_radians()))
            elif action == "Task":
                orientation.disabled, direction.disabled = value == "Plane", value == "Pose"
            elif action == "Apply target":
                active_task = task_name.lower()
                active_pose = np.r_[xyz, vtf.SO3.from_rpy_radians(*np.deg2rad(rpy)).wxyz]
                reference = sim.q[:1].clone()
                p, _, jacobian = solver.geometry(reference)
                origin = p[0].cpu().numpy()
                if active_task == "plane":
                    normal = np.asarray(plane_direction, dtype=np.float64)
                    if not np.isfinite(normal).all() or np.linalg.norm(normal) == 0:
                        raise ValueError("The plane direction must be finite and nonzero.")
                    normal /= np.linalg.norm(normal)
                    active_pose[:3] -= normal * (normal @ (active_pose[:3] - origin))
                    tangent = np.cross(normal, np.eye(3)[np.argmin(np.abs(normal))])
                    tangent /= np.linalg.norm(tangent)
                    plane.position = origin
                    plane.wxyz = vtf.SO3.from_matrix(
                        np.column_stack((tangent, np.cross(normal, tangent), normal))
                    ).wxyz
                plane.visible = active_task == "plane"
                n = (
                    None
                    if active_task == "pose"
                    else torch.tensor(normal[None], device="cuda:0", dtype=torch.float32)
                )
                basis = (
                    solver.network(
                        reference, reference_geometry(jacobian.float(), active_task), active_task, n
                    )
                    .detach()
                    .double()
                )
                target = {
                    "position": torch.tensor(active_pose[None, :3], device="cuda:0"),
                    "rotation": torch.tensor(
                        vtf.SO3(active_pose[3:]).as_matrix()[None], device="cuda:0"
                    ),
                }
                outcomes, lines = {}, []
                for name, b in (
                    ("Learned coefficients", basis),
                    ("Full joints", torch.eye(7, device="cuda:0").expand(1, -1, -1)),
                ):
                    result = solver.solve(
                        reference,
                        b,
                        target,
                        active_task,
                        torch.zeros((1, b.shape[-1]), device="cuda:0"),
                    )
                    outcomes[name] = result
                    angular = (
                        f" / {result['angular_error'].item():.4f} rad"
                        if active_task == "pose"
                        else ""
                    )
                    lines.append(
                        f"**{name} ({b.shape[-1]} variables): "
                        f"{'success' if result['success'].item() else 'failed'}**"
                        f" · {result['position_error'].item():.5f} m{angular}"
                        f" · {result['seconds'] * 1000:.1f} ms"
                        f" · {result['iterations'].item()} iterations"
                        f" · status {result['status'].item()}"
                    )
                selected = outcomes[method_name]
                q_start, motion_step = sim.q.clone(), 0
                if selected["success"].item():
                    q_goal = selected["q"].float().expand_as(sim.q)
                    lines.append(
                        f"Applied **{method_name}**. Run / Pause controls physical tracking."
                    )
                else:
                    running, q_goal = False, q_start
                    lines.append(f"**{method_name} failed; paused without applying a solution.**")
                comparison.content = "\n\n".join(lines)
                marker.position, marker.wxyz = active_pose[:3], active_pose[3:]
            elif action == "Run / Pause":
                running = not running
            elif action == "Reset":
                sim.reset(sim.home)
                q_start, q_goal = sim.q.clone(), sim.home.expand_as(sim.q)
                running, motion_step, active_task = False, 0, "pose"
                active_pose = targets[0].copy()
                task.value, method.value, presets.value = "Pose", "Learned coefficients", "Target 1"
                position.value = tuple(active_pose[:3])
                orientation.value = tuple(np.rad2deg(vtf.SO3(active_pose[3:]).as_rpy_radians()))
                orientation.disabled, direction.disabled = False, True
                marker.position, marker.wxyz, plane.visible = (
                    active_pose[:3],
                    active_pose[3:],
                    False,
                )
                comparison.content = "Reset to home. Apply a target to compare both methods."
            elif action == "Quit":
                running, exiting = False, True

        # Interpolate the selected affine displacement through the physical actuators.
        if running:
            motion_step += 1
            phase = min(motion_step * 0.01 / experiment["motion_seconds"], 1.0)
            blend = phase**3 * (10 - 15 * phase + 6 * phase**2)
            sim.step(q_start + blend * (q_goal - q_start))
        if tick >= next_render:
            scene_view.update_from_arrays(
                sim.xpos[:1].cpu().numpy(), sim.xmat[:1].cpu().numpy().reshape(1, -1, 3, 3)
            )
            tool = sim.site_xpos[0, sim.site].cpu().numpy()
            rotation = vtf.SO3.from_matrix(sim.site_xmat[0, sim.site].cpu().numpy().reshape(3, 3))
            angular_error = np.linalg.norm((vtf.SO3(active_pose[3:]).inverse() @ rotation).log())
            extra = (
                f" / {angular_error:.4f} rad"
                if active_task == "pose"
                else f" · normal error {abs(normal @ (tool - origin)):.5f} m · orientation free"
            )
            telemetry.content = (
                f"**{'Running' if running else 'Paused'}** · {sim.num_envs} GPU worlds\n\n"
                f"Time: {sim.time[0].item():.2f} s · Preview: world 0\n\n"
                f"Tool xyz: {np.array2string(tool, precision=3)} m\n\n"
                f"Physical tracking error: {np.linalg.norm(tool - active_pose[:3]):.4f} m{extra}"
            )
            next_render = tick + 1 / 30
        delay = 0.01 - (time.perf_counter() - tick)
        if delay > 0:
            time.sleep(delay)
    server.stop()


if __name__ == "__main__":
    main()
