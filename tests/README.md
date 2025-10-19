# Test Suite Overview

This project keeps tests lean and split by scope so you can run the right level of coverage for the change you’re making.

## Layout

- `tests/unit/`: fast, in-process checks for individual helpers. They don’t touch the heavy dynamical models. Run these constantly while editing core utilities.
- `tests/integration/`: higher-level checks that exercise data pipelines and end-to-end behaviours. We split these into:
  - `test_reference_outputs.py`: quick sanity checks against frozen `.npz` fixtures.
  - `test_viability_regression.py`: marked with `@pytest.mark.slow`; recomputes viability data and control policies to catch regressions. These tests rely on the pre-generated fixtures under `tests/integration/data/`.

## Running

- Fast loop: `uv run pytest -k "not slow"` (or `pytest` if you’re outside uv). This runs unit tests and quick integration checks.
- Full regression: `uv run pytest -m slow`. Use this before large refactors or when touching models/control logic; they take minutes.
- Targeted files: append paths/keywords, e.g. `uv run pytest tests/unit/test_viability_utils.py`.

## Philosophy

- Prefer deterministic fixtures (`.npz`) for expensive computations. Regenerate them only when behaviour intentionally changes, and note the rationale in the PR/commit.
- Avoid duplicating logic inside tests; import the public functions and assert on shapes, types, or exact matches.
- Keep slow tests clearly marked and grouped so contributors (and future automation) can choose between fast feedback and deeper validation.
- When adding new tests, decide early whether they belong in the fast or slow bucket, document any new fixtures, and update this README if the workflow changes.
