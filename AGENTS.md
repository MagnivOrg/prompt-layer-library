# AGENTS.md

## Cursor Cloud specific instructions

This is a Python SDK library (`promptlayer`). There are no servers or databases to run.

### Quick reference

| Task | Command |
|------|---------|
| Install deps | `poetry install` |
| Lint | `make lint` |
| Tests | `poetry run pytest` |
| Tests (with `.env`) | `make test` |

### Gotchas

- **`make test` requires a `.env` file** — the Makefile's `test` target unconditionally sources `.env`. If the file doesn't exist the shell errors out. Use `poetry run pytest` directly instead, or create an empty `.env` file first.
- **`pre-commit` is not a Poetry dependency** — it must be installed separately inside the Poetry virtualenv: `poetry run pip install pre-commit`. This is needed before `make lint` will work.
- **Tests run fully offline** — VCR cassettes replay all HTTP calls. No API keys or network access are needed. The `pytest-network` plugin disables network by default.
- **Re-recording cassettes** requires `PROMPTLAYER_API_KEY`, `OPENAI_API_KEY`, and `ANTHROPIC_API_KEY` env vars, plus setting `PROMPTLAYER_IS_CASSETTE_RECORDING=true`.
