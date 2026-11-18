"""Command-line interface: runs the whole pipeline from the terminal."""
import argparse  # reads what you type in the terminal after the command name
import logging  # writes the log messages

import numpy as np

from penguin_pipeline import build_features, load_data, split_data, standardize

logger = logging.getLogger(__name__)  # this module's logger; __name__ is the name shown in the log line --> INFO penguin_pipeline.cli


def main() -> None:  # no arguments: main reads its options from the terminal. None: the result is the saved file, nothing is returned
    """Read the options, run the pipeline, save the arrays."""
    # No line in this file calls main(). The terminal command does, through the line
    # penguin-pipeline = "penguin_pipeline.cli:main" in pyproject.toml

    # 1. The options the command accepts
    # A. The parser is the object that knows the command's options. The description is the sentence shown by --help
    parser = argparse.ArgumentParser(description="Turn the Palmer Penguins CSV into model-ready NumPy arrays.")
    # B. Required positional argument. If it is missing, the command stops with:
    #    "the following arguments are required: csv_path"
    parser.add_argument("csv_path", help="path to the penguins CSV file")
    # C. With -- in front it is optional. If you don't write it, it takes its default value
    parser.add_argument("--output", default="penguins_data.npz", help="where to save the arrays (default: penguins_data.npz)")
    # D. The terminal always sends text, so these two options convert the value to float and int.
    #    With --seed sette (instead of 7) the conversion fails and argparse stops with a clear message
    parser.add_argument("--train-ratio", type=float, default=0.8, help="share of examples in the training set (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="random seed for the split (default: 42)")
    # E. Switches with no value: action="store_true" stores True if the option is present, False otherwise.
    #    --verbose: every detail, for when something doesn't add up. --quiet: only warnings and errors.
    #    Without these switches you would have to open the code and add print calls every time:
    #    with them the code stays the same and you decide from the terminal
    parser.add_argument("--verbose", action="store_true", help="also show debug messages")
    parser.add_argument("--quiet", action="store_true", help="only show warnings and errors")
    args = parser.parse_args()  # reads what you typed in the terminal

    # 2. The program decides which logs to show: this belongs here, not in the library.
    #    level is the threshold: messages below it stay hidden.
    #    With the INFO threshold, DEBUG messages are hidden: that is why "Dropped 11..." appears only with --verbose
    if args.verbose:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.WARNING
    else:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        # format is the shape of every line; each %(...)s is a placeholder that logging fills by itself:
        # asctime = date and time; levelname = INFO; name = penguin_pipeline.cli; message = Loading penguins.csv
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.train_ratio < 0.5:
        logger.warning("Training set smaller than test set (train ratio %s)", args.train_ratio)

    # 3. The pipeline, told through the logs
    # INFO message: %s is replaced by args.csv_path, which is why you read "Loading penguins.csv"
    logger.info("Loading %s", args.csv_path)
    # Read from the inside out:
    # A. load_data(args.csv_path) opens the CSV and returns a DataFrame with 344 rows
    # B. build_features drops the 11 incomplete rows and returns two arrays; logger.debug("Dropped 11...") runs in there
    # X, y = ... puts the first array in X (333 rows, 9 columns) and the second in y (333 labels)
    X, y = build_features(load_data(args.csv_path))
    # INFO message: the two %s are filled with the shapes of the arrays --> X (333, 9), y (333,)
    logger.info("Features ready: X %s, y %s", X.shape, y.shape)

    # split_data shuffles and splits into train and test. The terminal options come in here:
    # with the defaults (80/20, seed 42) you get 266 training rows and 67 test rows
    X_train, X_test, y_train, y_test = split_data(X, y, train_ratio=args.train_ratio, seed=args.seed)
    X_train, X_test, mu, std = standardize(X_train, X_test)
    # INFO message: %d for an integer (the seed) and two %s for the shapes
    logger.info("Split with seed %d: train %s, test %s", args.seed, X_train.shape, X_test.shape)

    # 4. Save all the arrays in one file, each under its own name:
    # A. the first argument is the file name: "penguins_data.npz", or the one passed with --output
    # B. each name=array saves one array under that name: that is why d.files shows ['X_train', 'X_test', ...]
    np.savez(args.output, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, mu=mu, std=std)
    # INFO message: %s is replaced by the file name
    logger.info("Saved to %s", args.output)