# Bank Churn Modeling
This is the final project of the "Machine Learning Engineer with Microsoft Azure" form Udacity. Here we use the Azure Machine Learning API, to programmatically run two types of automated learning methods. The first and most advanced one is AutoML which checks the quality of the provided data and runs different types of machine learning models to see which one gives the best result. The second method is Hyperparameter tuning where you write your own machine learning code, which accepts hyperparameters as input. The Azure API then optimizes these parameters by sampling the search space configured by the user.

After the model has been trained, it is deployed on an Azure webservice that takes json data as input and outputs the predicted result.

While the performed actions in these notebooks can also be performed in AzureML studio, it is faster and better for reproducibility to use a Jupiter notebook to perform these tasks using code. That sayd the studio environment is still very useful for exploring data and comparing multiple experiments.

This project required the student to find a custom dataset to use. So this specific project is about predicting bank churn.

### Table of contents
[Dataset](#Dataset)
[Automated ML](#Automated-ML)
[Hyperparameter Tuning](#Hyperparameter-Tuning)
[Model Deployment](#Model-Deployment)
[Screen Recording](#Screen-Recording)

## Dataset
This data set contains details of a bank's customers and the target variable is a binary variable reflecting the fact whether the customer left the bank (closed his/her account) or he/she continues to be a customer. The dataset contains 10.000 records and the label that specifies if a customer is churned is "Exited"

The data provided was balanced, didn't contain missing values and didn't have high cardinality features. This is verified by by running Azure autoML see [Data quality](#Data-quality)


### Overview
The dataset is available on Keggle https://www.kaggle.com/shivan118/churn-modeling-dataset

### Task
The task is a binary classification based on a number of numarical and one hot encoded values, the Scikit-learn LogisticRegression can ne used to train this model 

The features that where used are where the Geographical location is one hot encoded.

```
        "CreditScore"
        "Gender"
        "Age"
        "Tenure"
        "Balance"
        "NumOfProducts"
        "HasCrCard"
        "IsActiveMember"
        "EstimatedSalary"
        "Geography_France"
        "Geography_Germany"
        "Geography_Spain"
```        
        
### Access
Since the data had to come from an external source, and Keggle doesn't support direct linking to the dataset. Github was used to provide the data as raw CSV

```
        example_data = 'https://raw.githubusercontent.com/fuzzballb/nd00333-capstone/master/starter_file/cleaned_data.csv'
```

The dataset is registerd to the workspace by using the register method on the Tabular dataset object. The workspace attributes tells the method where to register the data and de key and description are also passed to be able to identify the dataset.

```
        key = "Bank-churn"
        description_text = "Bank churn DataSet for Udacity Course"
        ...
        dataset = Dataset.Tabular.from_delimited_files(example_data)        
        #Register Dataset in Workspace
        dataset = dataset.register(workspace=ws,
                                   name=key,
                                   description=description_text)
```

## Automated ML
First, autoML was configured to optimize for accuracy. and the task is set to do a classification. The column "Exited" is chosen as the label for the training, since that column specifies is a client has left the bank or still has an account. The number of cross validations is five.

For the training it is specified that an experiment should time-out if it takes longer then 30 minutes. Four itarations are allowed to run at simmultaniously

### Results
the results of the different modals used by automated ML are the following. In this case the VotingEnsemble gives the highest accuracy (0.8657). 

![best model](https://github.com/fuzzballb/nd00333-capstone/blob/master/Screenshots/AutoML/Best%20modal.PNG?raw=true "best model")

![widget](https://github.com/fuzzballb/nd00333-capstone/blob/master/Screenshots/AutoML/Widget.PNG?raw=true "widget")

#### Best fitted model 
The the best fitted model has the following properties

```
        ...
        max_leaf_nodes=None,
        max_samples=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        min_samples_leaf=0.01,
        min_samples_split=0.2442105263157895,
        min_weight_fraction_leaf=0.0,
        n_estimators=10,
        n_jobs=1,
        oob_score=False,
        random_state=None,
        verbose=0,
        warm_start=False
        ...
        weights=[0.2, 0.3, 0.2, 0.1, 0.1, 0.1]
```        
        
![registering model](https://github.com/fuzzballb/nd00333-capstone/blob/master/Screenshots/AutoML/Best%20model%20register.PNG?raw=true "registering model")        
        
![model registerd](https://github.com/fuzzballb/nd00333-capstone/blob/master/Screenshots/AutoML/RegisterdModel.PNG?raw=true "model registerd") 

#### Data quality 
The data provided was balanced, didn't contain missing values and didn't have high cardinality features 

```
        DATA GUARDRAILS: 

        TYPE:         Class balancing detection
        STATUS:       PASSED
        DESCRIPTION:  Your inputs were analyzed, and all classes are balanced in your training data.
                      Learn more about imbalanced data: https://aka.ms/AutomatedMLImbalancedData

        ****************************************************************************************************

        TYPE:         Missing feature values imputation
        STATUS:       PASSED
        DESCRIPTION:  No feature missing values were detected in the training data.
                      Learn more about missing value imputation: https://aka.ms/AutomatedMLFeaturization

        ****************************************************************************************************

        TYPE:         High cardinality feature detection
        STATUS:       PASSED
        DESCRIPTION:  Your inputs were analyzed, and no high cardinality features were detected.
                      Learn more about high cardinality feature handling: https://aka.ms/AutomatedMLFeaturization
```

#### AutoML Configuration
The hyper parameters that where used in this training run where

```
        experiment timeout minutes: 30
        max_concurrent_iterations:4
        iterations: 10
        primary_metric: accuracy
        task: classification
        n_cross_validations: 5
```

#### How to improve the model
These results can be improved by adding more itterations and cross validations to the AutoML method. 

## Hyperparameter Tuning

### Used Machine learning method 
The machine learning method that was used to determine if a client would leave the bank, is a LogisticRegression. With Logistic regression the outcome is ether true or false. In our case this is exectly what we are trying to predict.

### strategy used for finding optimal hyper parameters
The strategy used for finding the optimal hyper parameters is RandomParameterSampling. 

### Used hyperparameters
The paremeters that were used was a uniform value between 0.2 and 15 that determaind the "Inverse of regularization strength", whereby smaller values cause stronger regularization. this parameter is added to minimize the chance of overfitting. 

the other paremeter is the maximum number of iterations to converge. Here we specify this to be ether 100 or 300. 


### Results
The best Accuracy for the Hyperparameter tuning was: 0.812, using the folowing parameters

Inverse of regularization strength: 3.5888602469365685
Maximum amount of iterations: 100

For the Hyperparameter tuning, running tests with the other two available strategies (Grid sampling or Bayesian sampling) might also produce more optimal hyperparameters

![best model](https://github.com/fuzzballb/nd00333-capstone/blob/master/Screenshots/HyperParameters/Best%20model.PNG?raw=true "best model")

![widget](https://github.com/fuzzballb/nd00333-capstone/blob/master/Screenshots/HyperParameters/Rundetails.PNG?raw=true "widget")


## Model Deployment

### The deployment
First the best modal is saved as a static file (pkl). Then a new anaconda environment with a AciWebservice is created that runs score.py and has the pip packages specified in conda_env_v_1_0_0.yml. The score.py uses the saved modal to do the predictions. The data that can be sent to this webservice needs to be formatted to the JSON standaard, and the result will return as JSON as well

### inference
Since the published endpoint abites to the REST API standard, a HTML post with the following JSON body needs to be sent to get the predictions. In this case we are requesting two predictions.

```
        data = {"data":
                [
                  {
                    "CreditScore": 	502,
                    "Gender": 1,	
                    "Age": 42,	
                    "Tenure": 8,	
                    "Balance": 159660.8,	
                    "NumOfProducts":	3,
                    "HasCrCard":	1,
                    "IsActiveMember": 0,	
                    "EstimatedSalary": 113931.57,	
                    "Geography_France": 1,	
                    "Geography_Germany": 0,	
                    "Geography_Spain": 0
                  },
                  {
                    "CreditScore": 	543,
                    "Gender": 0,	
                    "Age": 22,	
                    "Tenure": 8,	
                    "Balance": 0,	
                    "NumOfProducts":	2,
                    "HasCrCard":	0,
                    "IsActiveMember": 0,	
                    "EstimatedSalary": 127587.22,	
                    "Geography_France": 1,	
                    "Geography_Germany": 0,	
                    "Geography_Spain": 0
                  }
              ]
            }
            
        inputJson = json.dumps(data)
```        

To authenticate with this endpoint and get the predictions, there needs to be a bearer token added to the header of the post. This token can be found in the endpoint details in Azure ML studio. The next step is the actual post, which gets a response containing the predictions in JSON format.

```
        import requests # Used for http post request
        api_key = '<API KEY>' # Replace this with the API key for the web service
        headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}
        response = requests.post(service.scoring_uri, inputJson, headers=headers)
        print(response.text)
```

the result is a JSON response

```
        "{\"result\": [1, 0]}"
```

## Screen Recording
See the following link (https://www.youtube.com/watch?v=wGHW-fCDeq8) for a screen recording containing [updated screencast]

- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

