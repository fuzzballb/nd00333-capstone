from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Workspace, Datastore, Dataset



def clean_data(data):
    # Clean and one hot encode data
    # x_df = data.to_pandas_dataframe().dropna()
    # x_df = data.to_pandas_dataframe().dropna()
    # Geographies = pd.get_dummies(x_df.Geography, prefix="Geography")
    # x_df.drop("Geography", inplace=True, axis=1)
    # x_df = x_df.join(Geographies)
    # x_df["Gender"] = x_df.Gender.apply(lambda s: 1 if s == "Female" else 0)

    return x_df
    

# TODO: Create TabularDataset using TabularDatasetFactory
# Data is located at:
# "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"
ds = TabularDatasetFactory.from_delimited_files("https://raw.githubusercontent.com/fuzzballb/nd00333-capstone/master/starter_file/cleaned_data.csv")

#x, y = clean_data(ds)
x= clean_data(ds)
y = x.pop("Exited")

# TODO: Split data into train and test sets.

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.20)

run = Run.get_context()    

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()
