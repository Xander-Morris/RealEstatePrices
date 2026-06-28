import os 
import tarfile 
import urllib.request   
import pandas as pd 
import matplotlib.pyplot as plt 

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

def _main():
    _fetch_housing_data()
    housing = _load_housing_data()
    print(housing.head())
    print(housing.info())
    print(housing.describe())
    housing.hist(bins=50, figsize=(10,5))
    plt.show()

if __name__ == "__main__":
    _main()