# Penguin Data Pipeline

A small, tested Python package that turns the raw Palmer Penguins CSV into model-ready NumPy arrays, from the command line or as a library.

## What it does

1. **Load** the CSV into a pandas DataFrame (344 rows).
2. **Drop incomplete rows**: any penguin with a missing measurement or a missing `sex` is removed (333 rows remain).
3. **One-hot encode** `island` and `sex`, and map the species to the labels 0, 1, 2 (Adelie, Chinstrap, Gentoo).
4. **Split** into a training set and a test set (80% / 20% by default), shuffled with a fixed seed so the split is reproducible.
5. **Standardize** every feature using the mean and standard deviation of the training set only.
6. **Save** all the arrays in a single `.npz` file.

## Installation

You need [uv](https://docs.astral.sh/uv/) and Git.

```bash
git clone https://github.com/vladimiroboccia1/penguin-data-pipeline.git
cd penguin-data-pipeline
uv sync
```

`uv sync` creates the virtual environment and installs the exact versions recorded in `uv.lock`.

Tested with Python 3.14.

## Command line

```bash
uv run penguin-pipeline penguins.csv
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--output` | `penguins_data.npz` | where to save the arrays |
| `--train-ratio` | `0.8` | share of examples in the training set |
| `--seed` | `42` | random seed for the split |
| `--verbose` | off | also show debug messages |
| `--quiet` | off | only show warnings and errors |

Example output:

```
2026-11-18 16:06:16,203 INFO penguin_pipeline.cli: Loading penguins.csv
2026-11-18 16:06:16,204 INFO penguin_pipeline.cli: Features ready: X (333, 9), y (333,)
2026-11-18 16:06:16,205 INFO penguin_pipeline.cli: Split with seed 42: train (266, 9), test (67, 9)
2026-11-18 16:06:16,205 INFO penguin_pipeline.cli: Saved to penguins_data.npz
```

The output file contains six named arrays:

| Array | Shape | Content |
| --- | --- | --- |
| `X_train` | (266, 9) | standardized training features |
|