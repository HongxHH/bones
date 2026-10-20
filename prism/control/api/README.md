# API layout

- `v1.py` — HTTP routes (catalog, sessions, runs, telemetry, workers)
- Domain logic lives outside the router:
  - `prism/control/runs.py` — Run lifecycle / prepare / measure-on-stop
  - `prism/control/attach.py` — player spawn (local or `PRISM_ATTACH_URL` agent)
  - `prism/control/security.py` — loopback attach gate
  - `prism/control/telemetry.py` — jsonl + metrics + CSV
  - `prism/control/experiment_spec.py` — ExperimentSpec schema
