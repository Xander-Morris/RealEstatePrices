"""
pipeline.py
===========

Builds the "assembly line" that cleans and prepares data before the model sees it.

A raw data row can't go straight into a model. It has missing values, text that
math can't chew on, and numbers on wildly different scales. A PIPELINE is a list
of machines (transformers) that the data passes through IN ORDER, each one fixing
one problem, until clean numbers come out the far end.

The huge win of a pipeline: you define these steps ONCE, and sklearn applies the
EXACT same steps to your training data, your test data, and any future data.
That kills a whole class of bugs where the test data gets prepared slightly
differently from the training data.
"""

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from transformers import CombinedAttributesAdder
from config import NUMERIC_COLUMNS, CATEGORICAL_COLUMN


def build_preprocessing_pipeline():
    """
    Return the full data-prep assembly line (not yet fitted to any data).

    It has two lanes running in parallel because numbers and text need totally
    different treatment:
      - the NUMBER lane cleans and scales the numeric columns,
      - the TEXT lane turns the one text column into numbers.
    """

    # ------------------------------------------------------------------ #
    # LANE 1: the numeric columns
    # ------------------------------------------------------------------ #
    # Pipeline runs these three machines in order on the number columns:
    num_pipeline = Pipeline([
        # 1) IMPUTER: fills in blank/missing cells. Some districts are missing
        #    'total_bedrooms'. A model can't handle a hole, so we plug each hole
        #    with the MEDIAN (middle value) of that column. Median beats the mean
        #    here because it isn't dragged around by a few giant outliers.
        ("imputer", SimpleImputer(strategy="median")),

        # 2) ATTRIBS_ADDER: our custom machine that invents the ratio columns
        #    (rooms_per_household, etc.). See transformers.py.
        ("attribs_adder", CombinedAttributesAdder()),

        # 3) SCALER: puts every column on the same scale. Right now 'population'
        #    is in the thousands while 'median_income' is around 3. Many models
        #    wrongly think "bigger number = more important." StandardScaler
        #    rescales each column to roughly the same spread so it's a fair fight.
        ("std_scaler", StandardScaler()),
    ])

    # ------------------------------------------------------------------ #
    # LANE 2 is defined inline below: the text column
    # ------------------------------------------------------------------ #
    # OneHotEncoder turns a text category into numbers WITHOUT inventing a fake
    # ranking. It makes one yes/no (1/0) column per category. So "INLAND"
    # becomes something like [0, 1, 0, 0, 0]. We do it this way instead of
    # "INLAND=1, NEAR BAY=2..." because those numbers would trick the model into
    # thinking NEAR BAY is "twice" INLAND, which is meaningless.

    # ------------------------------------------------------------------ #
    # COMBINE BOTH LANES
    # ------------------------------------------------------------------ #
    # ColumnTransformer says "send THESE columns down lane 1, THOSE down lane 2,"
    # then staples the results back together side by side into one clean grid.
    full_pipeline = ColumnTransformer([
        # (name, which machine, which columns to feed it)
        ("num", num_pipeline, NUMERIC_COLUMNS),
        ("cat", OneHotEncoder(), [CATEGORICAL_COLUMN]),
    ])

    return full_pipeline
