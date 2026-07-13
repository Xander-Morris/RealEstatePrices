"""
main.py
=======

The CONDUCTOR of the orchestra.

This file doesn't do any heavy lifting itself. It just calls the other files in
the right ORDER, like a recipe. Read this file top to bottom and you'll
understand the whole machine-learning workflow at a glance:

    1. GET the data.
    2. LOOK at the data (plots) so we understand it.
    3. SPLIT into a training pile and a locked test pile.
    4. BUILD the data-cleaning pipeline.
    5. TRAIN a model and honestly measure it (cross-validation).
    6. TUNE the model's settings (grid search).
    7. RANK the features and DROP the weak ones.
    8. GRADE the final model on the locked test set.

Each numbered step below matches that list.
"""

from data import (
    fetch_housing_data,
    load_housing_data,
    add_income_category,
    stratified_split,
)
from plots import (
    plot_all_histograms,
    plot_income_histogram,
    plot_geographic_scatter,
)
from pipeline import build_preprocessing_pipeline
from modeling import (
    separate_features_and_labels,
    train_forest,
    evaluate_on_training,
    cross_validate,
    grid_search_best_forest,
    get_feature_names,
    report_feature_importances,
    build_select_and_predict_pipeline,
    final_evaluation,
)


def main():
    # ------------------------------------------------------------------ #
    # STEP 1 — GET THE DATA
    # ------------------------------------------------------------------ #
    fetch_housing_data()                 # download + unzip it (once)
    housing = load_housing_data()        # read the CSV into a DataFrame

    # A couple of quick text summaries in the terminal so we know what we've got:
    print(housing.head())                # first 5 rows
    housing.info()                       # column names + how many blanks
    print(housing.describe())            # min/max/average of each number column

    # ------------------------------------------------------------------ #
    # STEP 2 — LOOK AT THE DATA (before we split, just to understand it)
    # ------------------------------------------------------------------ #
    # We add income buckets first because one of the plots uses them.
    housing = add_income_category(housing)
    plot_all_histograms(housing)         # shape of every numeric column
    plot_income_histogram(housing)       # are the income buckets balanced?

    # ------------------------------------------------------------------ #
    # STEP 3 — SPLIT into training and (locked) test piles
    # ------------------------------------------------------------------ #
    # After this line, we pretend the test set DOESN'T EXIST until the very end.
    strat_train_set, strat_test_set = stratified_split(housing)

    # A picture of just the training data on the California map.
    plot_geographic_scatter(strat_train_set.copy())

    # Separate each pile into inputs (X) and the answer (y).
    X_train, y_train = separate_features_and_labels(strat_train_set)
    X_test, y_test = separate_features_and_labels(strat_test_set)

    # ------------------------------------------------------------------ #
    # STEP 4 — BUILD the data-cleaning pipeline and RUN it on the training inputs
    # ------------------------------------------------------------------ #
    # fit_transform = learn the medians/scales from training data AND apply them.
    full_pipeline = build_preprocessing_pipeline()
    X_train_prepared = full_pipeline.fit_transform(X_train)

    # ------------------------------------------------------------------ #
    # STEP 5 — TRAIN a first model and measure it honestly
    # ------------------------------------------------------------------ #
    forest = train_forest(X_train_prepared, y_train)
    evaluate_on_training(forest, X_train_prepared, y_train)  # optimistic number
    cross_validate(forest, X_train_prepared, y_train)        # honest number

    # ------------------------------------------------------------------ #
    # STEP 6 — TUNE the model's settings to squeeze out more accuracy
    # ------------------------------------------------------------------ #
    best_forest = grid_search_best_forest(X_train_prepared, y_train)

    # ------------------------------------------------------------------ #
    # STEP 7 — RANK features, then build the "prep + DROP weak + predict" model
    # ------------------------------------------------------------------ #
    feature_names = get_feature_names(full_pipeline)
    importances = report_feature_importances(best_forest, feature_names)

    # This one pipeline cleans data, keeps only the strongest features, and
    # predicts — all in a single object we can reuse on any new data.
    final_model = build_select_and_predict_pipeline(
        full_pipeline, importances, best_forest
    )
    # Re-fit the whole chain on the raw training inputs so the feature-selection
    # step and the forest are trained together on the pruned feature set.
    final_model.fit(X_train, y_train)

    # ------------------------------------------------------------------ #
    # STEP 8 — GRADE the final model on the locked test set (the honest score)
    # ------------------------------------------------------------------ #
    final_evaluation(final_model, X_test, y_test)


# This guard means "only run main() if this file is launched directly."
# If some other file imports this one, main() will NOT auto-run. Standard Python.
if __name__ == "__main__":
    main()
