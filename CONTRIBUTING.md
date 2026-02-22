## Contributing

- Use Python 3.9+; create a virtualenv (`python -m venv .venv`) and install `requirements.txt`.
- Keep data out of git. If you need small samples for tests, place them in `data/sample/` and document them.
- Run scripts from the repo root so relative paths resolve; prefer `python src/.../script.py` over moving files around.
- When adding a pipeline step, write a short note in `docs/pipeline.md` and drop schema expectations in a test or comment.
- Naming: snake_case for files, avoid spaces, keep outputs in `data/interim/` or `data/processed/`.
- Formatting: no heavy dependencies—stick to PEP8-ish style; docstrings optional for scripts, but add inline comments where logic is non-obvious.
