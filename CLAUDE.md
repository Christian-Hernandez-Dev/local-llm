# Local LLM — project rules

Side project run with the `/side-project` playbook. Read `README.md` (Roadmap) and
`docs/decisions.md` at the start of each slice.

## Commands
- `make setup` / `make check` (ruff + pyright + pytest) / `make train` / `make generate`
- Always use the Makefile or `UV_PROJECT_ENVIRONMENT=.venv.nosync uv ...`. The venv must stay in `.venv.nosync`.

## Layout
- `src/local_llm/`: `data.py` (text) → `tokenizer.py` → `model.py` → `train.py` → `cli.py`
- `data.nosync/`, `checkpoints.nosync/`: downloaded text and trained weights (gitignored)

## Rules
- Repo lives in iCloud `~/Documents`: anything heavy goes in a `*.nosync` folder. The git dir is `.git.nosync`.
- The user is learning how LLMs work: explain each piece in plain language as it's built, defining new terms.
- Keep the model hand-written (no Hugging Face `transformers` model classes) until the optional fine-tune slice.
- Simple, deep modules, obvious names. Tests where bugs would hurt.
- Secrets only in `.env` (gitignored). The gitleaks pre-commit hook must pass.
- Branch + PR per slice, Conventional Commits, squash-merge. Never push to `main`, never merge.
