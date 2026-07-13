"""
data.py
=======

Everything about GETTING the data and SPLITTING it into two piles.

Two jobs live here:

1. Download the dataset from the internet and load it into memory.
2. Split it into a "training set" (the pile we learn from) and a "test set"
   (a locked box we only open at the very end to grade ourselves honestly).

Why split at all? Imagine studying for a test using the exact questions that
will be on the exam. You'd get 100% but you wouldn't actually know the subject.
ML models cheat the same way: if we test on data they already saw, the score is
a lie. So we hide some data in a "test set" and never let the model peek.
"""

import os
import tarfile
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from config import (
    HOUSING_PATH,
    HOUSING_URL,
    RANDOM_SEED,
    INCOME_CAT_COLUMN,
)


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """
    Download the data file from the internet and unzip it onto your hard drive.

    A .tgz file is like a shrunk-down cardboard box with a CSV inside. We:
      1. make sure the destination folder exists,
      2. download the box,
      3. open the box and take the CSV out.
    """
    # exist_ok=True means "make this folder, but don't yell at me if it's already
    # there." Without it, running twice would crash on the second run.
    os.makedirs(housing_path, exist_ok=True)

    # Where we'll save the downloaded box.
    tgz_path = os.path.join(housing_path, "housing.tgz")

    # Actually go to the URL and copy the file down to tgz_path.
    urllib.request.urlretrieve(housing_url, tgz_path)

    # Open the box and pour its contents into the folder.
    # filter="data" is a newer safety setting that blocks sneaky/malicious
    # entries inside the archive from writing outside our folder.
    with tarfile.open(tgz_path) as housing_tgz:
        housing_tgz.extractall(path=housing_path, filter="data")


def load_housing_data(housing_path=HOUSING_PATH):
    """
    Read the CSV file off the disk and hand it back as a pandas DataFrame.

    A "DataFrame" is just a spreadsheet that lives in Python's memory: rows and
    named columns you can slice, filter, and do math on.
    """
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def add_income_category(housing):
    """
    Add a temporary helper column that buckets districts by how rich they are.

    WHY WE DO THIS (this is the clever part):
    Median income is the single strongest predictor of house price. When we split
    into train/test, we want BOTH piles to contain a fair mix of poor, middle,
    and rich districts. If, by bad luck, all the rich districts landed in the
    test set, our model would look terrible for no good reason.

    So we chop income into 5 groups (1 = poorest ... 5 = richest) and later tell
    the splitter: "keep these 5 groups balanced across both piles." This is
    called STRATIFIED sampling ("strata" = layers).

    pd.cut() is like sorting people into height-based lines:
      bins    = the fence posts (income boundaries) where one group ends and the
                next begins. np.inf just means "and everything above."
      labels  = the name we give each bucket (1 through 5).
    """
    housing[INCOME_CAT_COLUMN] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    return housing


def stratified_split(housing):
    """
    Split the data into (train, test) while keeping income groups balanced.

    Returns two DataFrames: the training pile and the test pile.
    """
    # test_size=0.2 means 20% goes into the locked test box, 80% is for learning.
    # n_splits=1 means "just give me one split, please" (not several).
    # random_state makes the shuffle repeatable (see RANDOM_SEED in config.py).
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=RANDOM_SEED
    )

    # .split() hands back the ROW NUMBERS for each pile, not the rows themselves.
    # We ask it to balance based on the income_cat column we added earlier.
    # The loop runs exactly once (because n_splits=1); it's just how sklearn
    # hands the indices back to us.
    for train_index, test_index in splitter.split(housing, housing[INCOME_CAT_COLUMN]):
        # .loc[row_numbers] grabs those specific rows out of the full table.
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    # The income_cat column was only a tool to help us split fairly. It has done
    # its job, so we remove it from BOTH piles to get back to the real columns.
    for subset in (strat_train_set, strat_test_set):
        subset.drop(INCOME_CAT_COLUMN, axis=1, inplace=True)  # axis=1 = "a column"

    return strat_train_set, strat_test_set
