"""
config.py
=========

Think of this file as the "settings box" for the whole project.

Instead of typing the same file paths, web links, and magic numbers over and
over inside every other file (and then crying when we need to change one of them
in 12 different places), we write them down ONCE here. Every other file just
asks this file: "hey, what's the path again?"

Rule of thumb you'll hear forever: "Don't repeat yourself" (DRY). This file is
that rule in action.
"""

import os

# --------------------------------------------------------------------------- #
# WHERE THE DATA LIVES
# --------------------------------------------------------------------------- #

# This is the website folder where the practice dataset is stored online.
# (It's the famous "California housing" dataset from a popular ML textbook.)
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"

# os.path.join glues folder names together using the correct slash for YOUR
# computer. On Windows that's "\", on Mac/Linux it's "/". Letting Python do it
# means our code works everywhere without us worrying about slashes.
# Result here: "datasets/housing"
HOUSING_PATH = os.path.join("datasets", "housing")

# The full web address of the compressed (.tgz = a zipped-up bundle) data file.
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


# --------------------------------------------------------------------------- #
# THE "RANDOM SEED"
# --------------------------------------------------------------------------- #

# Lots of ML steps use randomness (like shuffling cards before dealing).
# A "seed" is like telling the random-number machine: "start from THIS exact
# spot every time." That way we get the SAME shuffle every run, so results are
# reproducible and you can trust that a change in the score came from YOUR code,
# not from luck. 42 is a traditional joke-number people use; any number works.
RANDOM_SEED = 42


# --------------------------------------------------------------------------- #
# COLUMN NAMES (the labels at the top of each spreadsheet column)
# --------------------------------------------------------------------------- #

# The thing we are trying to PREDICT. In ML we call this the "target" or "label".
# Here: the median house value (price) for a district.
TARGET_COLUMN = "median_house_value"

# The one column that is TEXT instead of numbers ("NEAR BAY", "INLAND", ...).
# Text columns need special handling, so we keep its name handy.
CATEGORICAL_COLUMN = "ocean_proximity"

# A helper column we invent later to split the data fairly (explained in data.py).
# It's not a real feature we predict with, so we remember to throw it away.
INCOME_CAT_COLUMN = "income_cat"

# The numeric feature columns, IN THE EXACT ORDER they appear in the table
# AFTER we drop the target and the text column.
#
# Why does the order matter so much? Our custom "attribute adder" (see
# transformers.py) grabs columns BY POSITION (column #3, column #4, ...), not by
# name. So if this order is wrong, it will do math on the wrong columns and give
# nonsense. Writing the order here, once, keeps everyone in sync.
NUMERIC_COLUMNS = [
    "longitude",          # position 0
    "latitude",           # position 1
    "housing_median_age", # position 2
    "total_rooms",        # position 3  <-- used by the adder
    "total_bedrooms",     # position 4  <-- used by the adder
    "population",          # position 5  <-- used by the adder
    "households",          # position 6  <-- used by the adder
    "median_income",       # position 7
]
