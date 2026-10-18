# Penguin Data Pipeline

A reusable Python pipeline that turns the raw Palmer Penguins dataset into model-ready NumPy arrays.

## What it does

1. **Load** the CSV into a pandas DataFrame
2. **Build features**: drop incomplete rows, one-hot encode `island` and `sex`, map species to labels 0, 1, 2
3. **Split** into a training set (80%) and a test set (20%), randomly and reproducibly
4. **Standardize** every feature using the mean and standard deviation of the training set only

## Result

| Array | Shape |
| --- | --- |
| `X_train` | (266, 9) |
| `X_test` | (67, 9) |
| `y_train` | (266,) |
| `y_test` | (67,) |

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
Tested with Python 3.14.
```

Then open `demo.ipynb` and run all cells.

## Project structure

- `penguin_pipeline.py`: the pipeline functions
- `demo.ipynb`: example usage and plots
- `penguins.csv`: the dataset

## Design choices

- Why one-hot encoding for `island` and `sex`: models only work with numbers, but mapping the islands to 0, 1, 2 would invent an order and a distance between them (Dream would sit "between" Biscoe and Torgersen). One-hot encoding gives each category its own 0/1 column, so no false relationship is introduced.
- Why mean and standard deviation come from the training set only: the test set must behave like completely new data. Computing these statistics on all the data would leak information from the test set into training (data leakage) and make the evaluation look better than it really is. In production, new data is scaled with the statistics saved from training.

## Data

Palmer Archipelago (Antarctica) penguin data, collected by Dr. Kristen Gorman and the Palmer Station LTER, available under a CC-0 license.
Horst AM, Hill AP, Gorman KB (2020). palmerpenguins: Palmer Archipelago (Antarctica) penguin data. doi:10.5281/zenodo.3960218

## Next steps

Train a classifier on these arrays.