# FR3 GPU simulation: setup and validation

The current UI compares learned GPU coefficient IK with full-joint GPU IK and drives the selected successful result. The original preview measurements below describe the earlier UI; current controls, tolerances, and comparison results are in [GPU coefficient IK](coefficient_ik.md).

Validated on 2026-10-31 in C:\Users\Victor\Documents\Projects\MetaArm.

## Launch

From the project root in PowerShell:

~~~powershell
uv run --locked python -m metaarm
~~~

Open [the local experiment](http://127.0.0.1:8080). It starts paused with 64 GPU worlds and displays world zero. Choose a preset or edit the target, press **Apply target**, then **Run / Pause**. Reset restores the physical home state and clock; target inputs remain available for the next application. Quit shuts down the server.

The target frame represents the applied goal. Position inputs use metres; orientation inputs use roll/pitch/yaw degrees. Configuration poses use scalar-first unit quaternions. Minimize the control panel through its title bar to inspect the robot in a narrow browser.

The environment already exists. To recreate it after installing uv:

~~~powershell
uv sync --locked
~~~

Python 3.12.16 was installed with uv. PyTorch uses the official cu130 index; no separate system CUDA toolkit was needed. Cold GPU initialization took about 80 seconds due to compilation. Subsequent launches reuse the kernel cache.

## Validated stack

| Component | Observed value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4070, 12,282 MiB |
| Driver | 610.88 |
| Python / uv | 3.12.16 / 0.13.6 |
| MuJoCo / MuJoCo Warp | 3.17.0 / 3.17.0 |
| NVIDIA Warp | 1.18.0; toolkit 13.4, driver API 13.3 |
| PyTorch | 2.11.0+cu130; CUDA 13.0 |
| NumPy | 2.5.4 |
| mjviser / Viser | 0.0.14 / 1.1.1 |
| Menagerie commit | a662de603224c47123b0783821c58ad4402ad3e3 |

Exact transitive versions/hashes are in uv.lock; robot checksums and license are in assets/franka_fr3/.

## Physical model and controls

The scene contains the pinned FR3, a table, floor, and fixed contact box. MuJoCo model attachment preserves upstream mesh paths and original robot joint names. Base/world frames coincide. The tool is attachment_site; no gripper is added.

Reference IK searches seven joint variables with 0.0001 m position and 0.001 rad orientation tolerances. It controls the preview, not learned/reduced IK. Kinematic convergence does not guarantee collision-free motion.

GPU physics retains the original position actuators, gravity, implicit-fast integration, a 0.002 s physics step, and a 0.01 s control interval. Static actuator/gravity tracking error is distinct from IK tolerance: the second preset settled around 0.0052 m and 0.0112 rad error in the browser check.

## Verification

~~~powershell
uv run --locked pytest -q
uv run --locked ruff check src tests
uv run --locked ruff format --check src tests
~~~

Three tests passed. They check analytic Jacobians against central differences, configured IK targets, unreachable/invalid target and seed errors, shared GPU storage, CPU/GPU free motion and true box contact, independent worlds, complete repeatable resets, and invalid references.

CPU/GPU comparisons use absolute position/velocity tolerances of 0.001 rad and 0.02 rad/s over 0.3 s. These are short numerical checks, not general simulator-agreement bounds. Browser verification exercised robot rendering, run/pause, target changes/application, reset to time zero, and shutdown.

## Throughput and allocation observations

A warmed test advanced 200 control intervals with five physics steps plus a geometry refresh each. It retained state/reference validation and used CUDA graphs, without browser rendering or real-time pacing.

| Worlds | Mean control wall time | Physics world-steps/s | Peak Warp pool allocation |
| --- | --- | --- | --- |
| 1 | 2.171 ms | 2,303 | 13.97 MiB |
| 64 | 2.217 ms | 144,339 | 46.88 MiB |
| 256 | 2.628 ms | 487,100 | 147.19 MiB |

Throughput counts physics steps across all worlds per wall-clock second. Allocation peaks were measured in fresh processes using Warp's high-water counter; PyTorch-owned tensors added less than 0.02 MiB. These values exclude driver/context overhead, graph/external allocations, browser graphics, and unrelated GPU usage. They are not total peak device usage or guaranteed performance.

## Scope

The delivered foundation includes the locked environment, pinned assets, GPU simulation, full-joint reference IK, and experiment UI. General/planar collection is implemented; see [motion-data collection](data_collection.md). Neural basis training, reduced coefficient IK, and hardware integration remain governed by the research documents. No inactive training controls or alternate research method are wired into the UI.
