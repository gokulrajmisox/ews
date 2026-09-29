# Legacy and archived files

This folder contains historical or alternate implementations retained for reference only.

- The supported application is `backend.main:app`, launched with `python run.py --server`.
- The supported ML implementation lives under `ml/`.
- The supported dashboard lives under `frontend/`.
- Files here are not imported by the default FastAPI app, Docker image command, or CI suite.
- Legacy scripts may assume the repository root is the working directory and may require older dependencies or integrations.

The archived JSON files under `artifacts/` are retained for comparison with the current checked-in artifacts under `results/`.
