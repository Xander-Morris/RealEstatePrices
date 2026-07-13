"""
plots.py
========

All the drawing/graphing code lives here, out of the way.

Looking at pictures of your data ("exploratory data analysis") is a huge part of
ML. But plotting code is noisy and clutters the main logic, so we tuck every
chart into its own little function here. main.py just says "draw the scatter
plot" without caring HOW.
"""

import os

import matplotlib.pyplot as plt

from config import HOUSING_PATH


def display_current_plot(file_name, path=HOUSING_PATH):
    """
    Either SAVE the current chart to a file, or POP IT UP on screen.

    Some setups (like running on a server, or inside certain tools) have no
    screen to show a window on. matplotlib calls that a "headless"/"agg" backend.
    So we check: if there's no screen, save a .png; otherwise show the window.
    Either way we close the figure at the end so the next chart starts fresh and
    we don't leak memory by piling up open figures.
    """
    if "agg" in plt.get_backend().lower():
        # No screen available -> save a picture file instead.
        output_path = os.path.join(path, file_name)
        # dpi = dots per inch (sharpness). bbox_inches="tight" trims empty
        # whitespace around the edges.
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")
    else:
        # We have a screen -> open a window with the chart.
        plt.show()

    # Close it so we can happily make the next plot without leftovers.
    plt.close()


def plot_income_histogram(housing):
    """
    Draw a bar chart of how many districts fall into each income bucket (1-5).

    A histogram answers "how common is each value?" It's a quick sanity check
    that our income buckets aren't wildly lopsided before we split the data.
    """
    housing["income_cat"].hist()
    display_current_plot("income_categories.png")


def plot_all_histograms(housing):
    """
    Draw a histogram for EVERY numeric column at once.

    This is the first thing pros do with new data: one glance shows the range,
    the typical value, and weird stuff (like values that were capped at a max).
    """
    housing.hist(bins=50, figsize=(14, 10))  # bins=50 -> 50 bars per chart
    plt.tight_layout(pad=2.0)                 # add breathing room between charts
    display_current_plot("housing_histograms.png")


def plot_geographic_scatter(housing):
    """
    Draw California on a map and color each district by house price.

    x=longitude, y=latitude literally plots each district at its real-world spot,
    so the shape of California appears. Then:
      s (size of dot)   = population (bigger dot = more people),
      c (color of dot)  = median_house_value (the 'jet' colormap goes
                          blue=cheap -> red=expensive).
    You'll SEE that pricey districts hug the coast. That visual hunch is exactly
    why longitude/latitude end up being useful predictors.
    """
    housing.plot(
        kind="scatter",
        x="longitude",
        y="latitude",
        alpha=0.4,  # make dots see-through so dense clusters show up as darker
        s=housing["population"] / 100,
        label="population",
        figsize=(10, 7),
        c="median_house_value",
        cmap=plt.get_cmap("jet"),
        colorbar=True,  # show the price color-scale legend on the side
    )
    plt.legend()
    display_current_plot("scatter.png")
