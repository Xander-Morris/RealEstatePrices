"""
transformers.py
===============

Our own custom "machines" that reshape the data.

In scikit-learn, a TRANSFORMER is any object with a .fit() and a .transform()
method. Think of it as a machine on a factory conveyor belt: raw data goes in
one end, changed data comes out the other. sklearn gives us lots of these
(scalers, encoders...), but sometimes we need to build our own. That's what
lives here.

The two magic methods:
  fit(X)        -> the machine LOOKS at the data and remembers whatever it needs
                   (e.g. an average). It changes nothing; it just learns.
  transform(X)  -> the machine actually CHANGES the data using what it learned.

By writing our custom steps as transformers, we can drop them into a Pipeline
(see pipeline.py) and sklearn will call fit/transform for us at the right time.

We inherit from two sklearn helper classes:
  BaseEstimator   -> gives us get_params/set_params for free (needed by
                     GridSearch, which pokes at our settings).
  TransformerMixin-> gives us a free fit_transform() (= fit then transform).
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from config import NUMERIC_COLUMNS

# --------------------------------------------------------------------------- #
# Figure out WHICH column is which, by name, ONE time.
#
# The original code hard-coded "column 3, column 4, ..." as bare numbers. That's
# fragile: if the column order ever changes, those numbers silently point at the
# wrong data. Instead we look up the positions from the single list of column
# names in config.py. If the order changes, this updates automatically.
#
# .index("total_rooms") means "tell me the position of 'total_rooms' in the
# list" (counting from 0).
# --------------------------------------------------------------------------- #
rooms_ix = NUMERIC_COLUMNS.index("total_rooms")
bedrooms_ix = NUMERIC_COLUMNS.index("total_bedrooms")
population_ix = NUMERIC_COLUMNS.index("population")
households_ix = NUMERIC_COLUMNS.index("households")


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """
    Invents NEW, smarter columns by combining existing ones.

    Raw counts like "total_rooms" for a whole district aren't that meaningful on
    their own (a big district has more rooms just because it's big). Ratios are
    smarter:
      rooms_per_household     = rooms  / households   (roomy homes vs cramped?)
      population_per_household = people / households   (crowded homes?)
      bedrooms_per_room        = bedrooms / rooms      (bedroom-heavy layout?)

    In the feature-importance ranking, these invented ratios beat the raw counts
    they came from! That's why we can't just delete the raw columns: the ratios
    are BUILT from them. This machine adds the ratios as brand-new columns glued
    onto the right edge of the data.
    """

    def __init__(self, add_bedrooms_per_room=True):
        # A knob you can turn on/off. Because it's a settable option, GridSearch
        # can even TEST whether adding bedrooms_per_room helps or hurts.
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        # Nothing to learn here (no averages to memorize), so we just say "done".
        # We still MUST define fit() because Pipelines will call it.
        return self

    def transform(self, X):
        # X is a plain NumPy grid of numbers (rows x columns). We grab whole
        # columns by their position and divide them element-by-element.
        # X[:, rooms_ix] means "every row, but only the rooms column."
        # first index is row, second index is column in expressions like X[2, 3] when dealing with NumPy grids
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix] 
        population_per_household = X[:, population_ix] / X[:, households_ix]

        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            # np.c_[...] stacks columns side by side. Here: keep all original
            # columns (X) and tack the 3 new ratio columns onto the right.
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]

        # If the knob is off, add only the two ratios.
        return np.c_[X, rooms_per_household, population_per_household]


class TopFeatureSelector(BaseEstimator, TransformerMixin):
    """
    THROWS AWAY the weak columns and keeps only the strongest ones.

    This is the "remove unimportant attributes" step you asked for.

    How it works, in kid terms: after training a model once, the model can tell
    us an "importance score" for every column (how much it leaned on that column
    to make predictions). We sort those scores, pick the top K winners, and this
    machine simply keeps those K columns and drops the rest.

    Why do it as a transformer instead of deleting columns by hand?
      * The important/unimportant columns include INVENTED ones (like
        bedrooms_per_room) that don't exist until the pipeline builds them. By
        selecting AFTER the pipeline creates all columns, we can drop weak ones
        no matter where they came from.
      * It plugs into the pipeline, so the SAME dropping happens automatically to
        the training data, the test data, and any future data. No mismatches.

    (sklearn has a built-in called SelectFromModel that does something similar
    automatically. We hand-roll it here so you can SEE exactly what happens.)
    """

    def __init__(self, feature_importances, k):
        # feature_importances = the list of scores, one per column.
        # k = how many of the best columns to keep.
        self.feature_importances = feature_importances
        self.k = k

    def fit(self, X, y=None):
        # Work out WHICH column positions are the top k, and remember them.
        self.feature_indices_ = _indices_of_top_k(self.feature_importances, self.k)
        return self

    def transform(self, X):
        # Keep only the winning columns; every row, but just those column indices.
        return X[:, self.feature_indices_]


def _indices_of_top_k(feature_importances, k):
    """
    Return the column positions of the k highest importance scores.

    np.argsort sorts and gives back POSITIONS (indices), smallest score first.
    We take the last k of those (the biggest k), i.e. the strongest columns.
    np.sort at the end just puts those positions back in left-to-right order so
    the surviving columns stay in their original arrangement.
    """
    top_k = np.sort(np.argsort(np.array(feature_importances))[-k:])
    return top_k
