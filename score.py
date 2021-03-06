# +
# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib

from azureml.core.model import Model

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType

input_sample = pd.DataFrame(
    {
        "CreditScore": pd.Series([0.0], dtype="float64"), 
        "Gender": pd.Series([0.0], dtype="float64"), 
        "Age": pd.Series([0.0], dtype="float64"), 
        "Balance": pd.Series([0.0], dtype="float64"), 
        "NumOfProducts": pd.Series([0.0], dtype="float64"), 
        "HasCrCard": pd.Series([0.0], dtype="float64"), 
        "IsActiveMember": pd.Series([0.0], dtype="float64"), 
        "EstimatedSalary": pd.Series([0.0], dtype="float64"), 
        "Geography_France": pd.Series([0.0], dtype="float64"), 
        "Geography_Germany": pd.Series([0.0], dtype="float64"), 
        "Geography_Spain": pd.Series([0.0], dtype="float64"), 
    }
)
output_sample = np.array([0])

def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'automl_test_experiment_best_model2.pkl')
    print(model_path)
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise

@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
