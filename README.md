# Real Estate Prices — Learning ML with the California Housing dataset

A beginner-friendly walk through a full machine-learning workflow: predict the
median house price of a California district from things like income, location,
and rooms per household.

The code is heavily commented (explained like you're new to all of this). Read
the files in the order below and you'll learn the whole pipeline.

## How to run

```bash
pip install -r requirements.txt
cd src
python main.py
```

Charts are saved as `.png` files under `src/datasets/housing/` (or pop up in a
window if your setup has a screen).

## What each file does (read in this order)

| File | Its one job |
|------|-------------|
| `main.py` | The conductor. Calls everything in order — read this first for the big picture. |
| `config.py` | The settings box: paths, the web link, the random seed, column names. |
| `data.py` | Download the data, load it, and split it fairly into train/test piles. |
| `plots.py` | All the charts, tucked out of the way. |
| `transformers.py` | Our custom data machines: the ratio-column **adder** and the weak-feature **dropper**. |
| `pipeline.py` | Assembles the data-cleaning "assembly line". |
| `modeling.py` | Train, measure honestly, tune, rank features, and grade on the test set. |

## The workflow (matches the steps in `main.py`)

1. **Get** the data.
2. **Look** at it with plots so we understand it.
3. **Split** into a training pile and a *locked* test pile (no peeking!).
4. **Build** the cleaning pipeline (fill blanks, add ratio columns, scale, one-hot the text).
5. **Train** a Random Forest and measure it honestly with cross-validation.
6. **Tune** the model's settings with a grid search.
7. **Rank** every feature by importance and **drop** the weak ones (`TopFeatureSelector`).
8. **Grade** the final model on the locked test set — the one honest score.

## The "drop unimportant attributes" part

The forest tells us an *importance score* for every column. Weak columns (raw
counts like `total_rooms`, and rare categories like `ISLAND`) score near zero.
The `TopFeatureSelector` in `transformers.py` keeps only the top `TOP_K_FEATURES`
(set in `modeling.py`) and drops the rest — **inside** the pipeline, so training,
test, and future data all get pruned identically.

Want to experiment? Change `TOP_K_FEATURES` in `modeling.py` and watch the final
RMSE. That's the accuracy-vs-simplicity trade-off you're learning to feel out.
