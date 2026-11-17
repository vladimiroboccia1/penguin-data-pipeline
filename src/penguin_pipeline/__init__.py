"""penguin_pipeline: turns the Palmer Penguins CSV into model-ready NumPy arrays."""
from penguin_pipeline.pipeline import FEATURES, LABELS, build_features, load_data, split_data, standardize
