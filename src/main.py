import os 
import tarfile 
import urllib.request   
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler
from classes.combined_attributes_adder import CombinedAttributesAdder 

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def _load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

def _fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    with tarfile.open(tgz_path) as housing_tgz:
        housing_tgz.extractall(path=housing_path, filter="data")

def _display_current_plot(file_name, path=HOUSING_PATH):
    if "agg" in plt.get_backend().lower():
        output_path = os.path.join(path, file_name)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")
    else:
        plt.show()
    plt.close() # i do this here to ensure that i can make as many plots as i want without manually closing each time i call this function

def _main():
    _fetch_housing_data()
    housing = _load_housing_data()
    print(housing.head())
    housing.info()
    print(housing.describe())
    housing["income_cat"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
    housing.hist(bins=50, figsize=(14, 10))
    plt.tight_layout(pad=2.0)
    _display_current_plot("housing_histograms.png")
    housing["income_cat"].hist()
    _display_current_plot("income_categories.png")
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        print("NEW ITER")
        print(train_index)
        print(test_index)
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
        housing = strat_train_set.copy()
        housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                s=housing["population"]/100, label="population", figsize=(10,7),
                c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
        plt.legend()
        _display_current_plot("scatter.png")
        housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
        housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
        housing["population_per_household"] = housing["population"]/housing["households"]
        numeric_housing = housing.select_dtypes(include=[np.number])
        corr_matrix = numeric_housing.corr()
        print(corr_matrix["median_house_value"].sort_values(ascending=False))
        imputer = SimpleImputer(strategy="median")
        housing_num = housing.drop("ocean_proximity", axis=1)
        imputer.fit(housing_num)
        X = imputer.transform(housing_num)
        housing_tr = pd.DataFrame(
            np.asarray(X),
            columns=housing_num.columns.to_list(),
            index=housing_num.index,
        )
        cat_encoder = OneHotEncoder()
        housing_cat = housing[["ocean_proximity"]]
        housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
        print(housing_cat_1hot)
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler())
        ])
        housing_num_tr = num_pipeline.fit_transform(housing_num)
        print("TR: ", housing_num_tr)

if __name__ == "__main__":
    _main()
