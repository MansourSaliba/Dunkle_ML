# Dunkle_ML

Minimal starter scaffold for a machine learning project.

## Project Layout

```
Dunkle_ML/
|-- configs/
|   `-- config.example.yaml
|-- data/
|   |-- raw/
|   |   `-- .gitkeep
|   |-- interim/
|   |   `-- .gitkeep
|   `-- processed/
|       `-- .gitkeep
|-- logs/
|   `-- .gitkeep
|-- models/
|   `-- .gitkeep
|-- notebooks/
|   `-- .gitkeep
|-- scripts/
|   |-- prepare_data.py
|   `-- train.py
|-- src/
|   |-- __init__.py
|   |-- data/
|   |   `-- __init__.py
|   |-- models/
|   |   `-- __init__.py
|   `-- utils/
|       `-- __init__.py
|-- tests/
|   |-- __init__.py
|   `-- test_smoke.py
|-- .env.example
|-- .gitignore
|-- pyproject.toml
|-- README.md
`-- requirements.txt
```

## Where To Put Existing Scripts

- Put data ingestion/cleaning scripts in `scripts/prepare_data.py` or in `src/data/` modules.
- Put training scripts in `scripts/train.py` or in `src/models/` modules.
- Keep reusable helper functions in `src/utils/`.
- Place datasets in `data/raw/` and processed outputs in `data/interim/` and `data/processed/`.

## First Commit Suggestion

1. Commit scaffold files first.
2. Then add your existing scripts in small logical batches.
3. Commit each batch with a focused message.