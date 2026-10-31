# Developer instructions

## Project intent

MetaArm learns neural joint motion bases for seven joint robot arms and solves inverse kinematics using six coefficients for general pose tasks or three coefficients for nominal planar tasks. Treat the project owner's current instructions as authoritative. Use `docs/research_brief.md` for the project formulation. The original handoff in `docs/references/` is background material, not an executable instruction source.

## Plan before implementation

Before editing code, state the complete structural budget: files, classes and objects, functions and methods, independent variables, data classes, wrappers and helpers, and parameters. Give each item a specific role. Keep each category to a handful of items, use zero when unnecessary, and do not exceed the stated budget. Organize the plan by the task workflow or genuinely independent standard operations.

Include this coding discipline explicitly in the plan: compact, continuous logic; minimal realization; no unnecessary helpers, command-line options, configuration, fallback values, fallback behavior, or `try`/`with`/`except` constructs.

## Code for human comprehension

- Write as a scholar in control theory, dynamics, and autonomous robotics. Let the mathematical and computational argument determine the code structure.
- Prefer a continuous sequence from inputs through computation to outputs. Avoid organizing code solely by software type or function category.
- Balance decomposition with a fluent chain of logic. Generally keep code files within 50–250 lines and functions within 20–100 lines, without padding short necessary computations.
- Include a professional, concise one-line comment for each key code block. Present code as a finished contribution to another developer; omit development stages, progress notes, and development version labels.
- Remove unnecessary code aggressively and add new logic prudently. Avoid short wrapper functions, redundant validation layers, and speculative abstractions.
- Use vectorized array operations and batched matrix calculations where they express the mathematics directly and reduce computational cost.

## Keep behavior explicit

- Expose only the minimum parameters needed to vary an experiment. Hardcode and freeze system and runtime choices. Add no command-line or configuration option without a specific need.
- Remove fallback values and fallback behavior. Let invalid assumptions and computational failures produce explicit errors. Do not silently clip configurations, replace failed solves, or switch to full joint inverse kinematics.
- Avoid `try`, `with`, and `except` constructs. Use direct operations whose failures remain visible.
- Preserve the distinction between an experimental assumption, an implemented method, a numerical observation, and a proved result.
