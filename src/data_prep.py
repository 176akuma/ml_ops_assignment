import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
import joblib

import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://localhost:5000")

def fetch_housing_data():
  return pd.read_csv("/home/Python_Ws/Mlops/mlops-pipeline/raw_data/housing.csv")

def set_income_category(housing_selected):
    # set income category based on median income
    housing_selected["income_cat"] = pd.cut(housing_selected["median_income"], 
                                            bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                            labels=[1, 2, 3, 4, 5])
    return housing_selected

def get_strat_train_test_dataset(housing_selected):
    # stratified sampling
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    housing_split = split.split(housing_selected, housing_selected["income_cat"])
    # get train and test dataset
    for train_index, test_index in housing_split:
        train_set = housing_selected.loc[train_index]
        test_set = housing_selected.loc[test_index]
        
    # drop income_category from train and test dataset
    for set_ in (train_set, test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    
    return train_set, test_set

def sprt_train_and_label_set(train_set):
    # drop median_house_value from training data
    housing_tr = train_set.drop("median_house_value", axis=1)
    # create a new dataframe with median_house_value
    housing_labels = train_set["median_house_value"].copy()
    return housing_tr, housing_labels

def get_rmse(housing_labels, predicted_data):
    # get mean squared error to analyse prediction error
    mse = mean_squared_error(housing_labels, predicted_data)
    rmse = np.sqrt(mse)
    return rmse

# get housing data
housing = fetch_housing_data()
# copy median_income and median_house_value
housing_selected = housing[['median_income', 'median_house_value']].copy()
# set income category based on median_icome
housing_selected = set_income_category(housing_selected)
# stratified sampling
train_set, test_set = get_strat_train_test_dataset(housing_selected)
# seperate label and data from training set
housing_tr, housing_labels = sprt_train_and_label_set(train_set)
print("----------")


from sklearn.linear_model import LinearRegression
# linear regression model for best fit

with mlflow.start_run(run_name="Linear_regression"):
    # Log parameters
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("epochs", 100)


    model = LinearRegression()
    model.fit(housing_tr, housing_labels)

    # sample data to test from training set
    sample_data = housing_tr.iloc[:5]
    sample_labels = housing_labels.iloc[:5]

    # predict the median_house_value
    predicted_data = model.predict(sample_data)
    print("Predicted Price:", predicted_data)
    print("Actual Price:", list(sample_labels))

    # save the model
    model_filename = "linear_regression.joblib"
    joblib.dump(model, model_filename)

    # Log metrics
    mlflow.log_metric("r2_score", model.score(sample_data, sample_labels))

    # Log the model
    mlflow.sklearn.log_model(model, "linear_regression_model")

    print("MLflow run completed and data logged to the local server.")