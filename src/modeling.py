"""
modeling.py
===========

Where we actually TRAIN a model, measure how good it is, tune it, figure out
which columns matter, DROP the useless ones, and finally grade ourselves on the
locked test set.

Vocabulary you'll keep meeting:
  X          = the input columns (the "features": income, location, rooms...).
  y / labels = the answer we want to predict (the house price).
  fit        = "learn from this data" (a.k.a. "train").
  predict    = "now guess the answers for this data".
  RMSE       = Root Mean Squared Error: on average, how many DOLLARS off are our
               price guesses? Lower is better. It's in the same units as price,
               so "RMSE = 50000" means "typically wrong by about $50,000."
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV

from config import (
    TARGET_COLUMN,
    CATEGORICAL_COLUMN,
    NUMERIC_COLUMNS,
    RANDOM_SEED,
)
from pipeline import build_preprocessing_pipeline
from transformers import TopFeatureSelector


# How many of the strongest features to keep when we prune. From the importance
# ranking, a big chunk of the total importance sits in the top ~8 features and
# the tail (raw counts + rare ocean categories like ISLAND) barely matters.
# Try changing this number and watch the RMSE — that's how you learn the trade-off.
TOP_K_FEATURES = 8


def separate_features_and_labels(data_set):
    """
    Split one table into (X = inputs, y = the answer column).

    We drop the target column out of X because letting the model see the answer
    while training would be cheating. y is just that answer column on its own.
    """
    X = data_set.drop(TARGET_COLUMN, axis=1)   # everything EXCEPT the price
    y = data_set[TARGET_COLUMN].copy()         # ONLY the price
    return X, y


def print_scores(scores):
    """Pretty-print the results of cross-validation (explained below)."""
    print("Scores:", scores)
    print("Mean:", scores.mean())              # average error across the folds
    print("Standard deviation:", scores.std()) # how much the error wobbles


def train_forest(X_prepared, y):
    """
    Train a Random Forest on already-prepared (cleaned & scaled) data.

    A DECISION TREE asks yes/no questions ("is income > 4?") to funnel a district
    down to a price guess. A single tree tends to over-memorize the training
    data. A RANDOM FOREST grows MANY slightly-different trees and averages them,
    like asking a crowd instead of one person — usually much more accurate and
    less jumpy.
    """
    forest = RandomForestRegressor(random_state=RANDOM_SEED)
    forest.fit(X_prepared, y)
    return forest


def evaluate_on_training(forest, X_prepared, y):
    """
    Quick 'error on the data we trained on'. Useful, but READ THE WARNING.

    This number is almost always TOO GOOD, because the model has already seen
    these exact rows. It's like grading your homework with the answer key open.
    We print it only to compare against the honest cross-validation number next.
    """
    predictions = forest.predict(X_prepared)
    mse = mean_squared_error(y, predictions)   # average squared error
    rmse = np.sqrt(mse)                        # square-root to get back to $
    print("Training RMSE (optimistic!):", rmse)
    return rmse


def cross_validate(forest, X_prepared, y):
    """
    The HONEST way to estimate performance without touching the test set.

    K-FOLD CROSS-VALIDATION (here cv=10):
      1. Chop the training data into 10 equal chunks ("folds").
      2. Train on 9 chunks, test on the 1 held-out chunk. Record the error.
      3. Repeat 10 times, each time holding out a different chunk.
    Now we have 10 honest error scores (the model never graded itself on data it
    trained on in that round). Their average is a trustworthy estimate, and their
    spread tells us how reliable that estimate is.

    Note: sklearn maximizes scores, but for error we want to MINIMIZE. Its trick
    is to hand back NEGATIVE error ("neg_mean_squared_error"), so we flip the
    sign back with the minus before square-rooting.
    """
    scores = cross_val_score(
        forest, X_prepared, y, scoring="neg_mean_squared_error", cv=10
    )
    rmse_scores = np.sqrt(-scores)
    print_scores(rmse_scores)
    return rmse_scores


def grid_search_best_forest(X_prepared, y):
    """
    Automatically try many model SETTINGS and keep the best combo.

    A model has "hyperparameters" — dials you set BEFORE training (not learned
    from data). For a forest, examples:
      n_estimators = how many trees in the forest,
      max_features = how many columns each tree is allowed to consider.
    We don't know the best dial settings up front, so GridSearchCV brute-forces
    every combination in our grid, cross-validates each, and returns the winner.

    The two dictionaries below are two separate 'grids' of combos to try:
      - grid 1 tries 3x4 = 12 combos of trees x features,
      - grid 2 additionally turns off 'bootstrap' and tries 2x3 = 6 more.
    cv=5 means each combo is judged with 5-fold cross-validation.
    """
    param_grid = [
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]

    forest = RandomForestRegressor(random_state=RANDOM_SEED)
    grid_search = GridSearchCV(
        forest,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(X_prepared, y)

    print("Best hyperparameters found:", grid_search.best_params_)
    # .best_estimator_ is the fully-trained forest using the winning settings.
    return grid_search.best_estimator_


def get_feature_names(full_pipeline):
    """
    Build the list of column names AFTER the pipeline transformed the data.

    Why this is fiddly: the pipeline INVENTS columns (the ratio features) and
    EXPLODES the text column into several yes/no columns. So the prepared data
    has more columns than we started with, in this order:
        [ the 8 original numeric columns ]
      + [ the 3 invented ratio columns ]
      + [ one column per ocean_proximity category ]
    We must line these names up with the importance scores so we know which score
    belongs to which feature.
    """
    # Names of the invented ratio columns, in the exact order the adder appends
    # them (see CombinedAttributesAdder.transform).
    extra_attribs = ["rooms_per_household", "population_per_household", "bedrooms_per_room"]

    # Ask the fitted OneHotEncoder which categories it found (INLAND, ISLAND...).
    cat_encoder = full_pipeline.named_transformers_["cat"]
    cat_one_hot_attribs = list(cat_encoder.categories_[0])

    return NUMERIC_COLUMNS + extra_attribs + cat_one_hot_attribs


def report_feature_importances(best_forest, feature_names):
    """
    Print every feature next to its importance score, strongest first.

    'feature_importances_' is a list of numbers that add up to 1.0 — each is the
    share of the model's decision-making that leaned on that feature. Big number
    = the model relied on it a lot; near-zero = the model barely used it (a prime
    candidate to drop). Returns the raw importance list for the selector to use.
    """
    importances = best_forest.feature_importances_

    # zip() pairs each score with its name; sorted(..., reverse=True) ranks them
    # from most to least important.
    ranked = sorted(zip(importances, feature_names), reverse=True)
    print("\nFeature importances (most to least important):")
    for score, name in ranked:
        print(f"  {score:.5f}  {name}")

    return importances


def build_select_and_predict_pipeline(full_pipeline, importances, best_forest):
    """
    Chain EVERYTHING into one machine: prep -> drop weak features -> predict.

    This single pipeline is the payoff of all our work. Hand it a raw district
    and it will, in one call:
      1. clean + scale + add ratio columns + one-hot the text (full_pipeline),
      2. keep ONLY the TOP_K_FEATURES strongest columns (TopFeatureSelector),
      3. predict the price with the tuned forest.
    Because dropping happens INSIDE the pipeline, the exact same columns get
    dropped for training data, test data, and any future data — no mismatch bugs.
    """
    return Pipeline([
        ("preparation", full_pipeline),
        ("feature_selection", TopFeatureSelector(importances, TOP_K_FEATURES)),
        ("forest", best_forest),
    ])


def final_evaluation(model, X_test, y_test):
    """
    THE moment of truth: grade the finished model on the locked test set.

    We have not let the model peek at this data even once. Whatever RMSE we get
    here is our honest estimate of how the model will do on brand-new, real-world
    districts it has never seen. This is the number you'd actually report.
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    print("\nFINAL test-set RMSE (the honest score):", rmse)
    return rmse
