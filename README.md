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
| `X_test` | (67, 9) | standardized test features |
| `y_train` | (266,) | training labels |
| `y_test` | (67,) | test labels |
| `mu` | (9,) | mean of each feature, from the training set |
| `std` | (9,) | standard deviation of each feature, from the training set |

Load them back with NumPy:

```python
import numpy as np

data = np.load("penguins_data.npz")
X_train, y_train = data["X_train"], data["y_train"]
```

## As a library

```python
from penguin_pipeline import load_data, build_features, split_data, standardize

X, y = build_features(load_data("penguins.csv"))
X_train, X_test, y_train, y_test = split_data(X, y, train_ratio=0.8, seed=42)
X_train, X_test, mu, std = standardize(X_train, X_test)
```

See `demo.ipynb` for a longer example with plots.

## Tests

```bash
uv run pytest
```

The 10 tests run on a small hand-written DataFrame, not on the full CSV. They check the shape of the features, that the split is reproducible and loses no example, that standardization uses the training statistics only, that no missing value survives, and that the labels are exactly 0, 1, 2.

## Project structure

- `src/penguin_pipeline/pipeline.py`: the pipeline functions
- `src/penguin_pipeline/cli.py`: the `penguin-pipeline` command
- `tests/test_pipeline.py`: the tests
- `demo.ipynb`: example usage and plots
- `penguins.csv`: the dataset
- `pyproject.toml` and `uv.lock`: project metadata and locked dependency versions

## Design choices

- Why one-hot encoding for `island` and `sex`: models only work with numbers, but mapping the islands to 0, 1, 2 would invent an order and a distance between them (Dream would sit "between" Biscoe and Torgersen). One-hot encoding gives each category its own 0/1 column, so no false relationship is introduced.
- Why mean and standard deviation come from the training set only: the test set must behave like completely new data. Computing these statistics on all the data would leak information from the test set into training (data leakage) and make the evaluation look better than it really is. In production, new data is scaled with the statistics saved from training.

## Data

Palmer Archipelago (Antarctica) penguin data, collected by Dr. Kristen Gorman and the Palmer Station LTER, available under a CC-0 license.
Horst AM, Hill AP, Gorman KB (2020). palmerpenguins: Palmer Archipelago (Antarctica) penguin data. doi:10.5281/zenodo.3960218

## Next steps

Train a classifier on these arrays.