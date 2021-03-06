# Bank Churn Modeling Dataset

This data set contains details of a bank's customers and the target variable is a binary variable reflecting the fact whether the customer left the bank (closed his account) or he continues to be a customer.

## Dataset

### Overview
The dataset is available on Keggle https://www.kaggle.com/shivan118/churn-modeling-dataset

### Task
The task is a binary classification based on a number of numarical and one hot encoded values, the Scikit-learn LogisticRegression can ne used to train this model 

The features that where used are where the Geographical location is one hot encoded.

        "CreditScore"
        "Gender"
        "Age"
        "Balance"
        "NumOfProducts"
        "HasCrCard"
        "IsActiveMember"
        "EstimatedSalary"
        "Geography_France"
        "Geography_Germany"
        "Geography_Spain"
        
### Access
Since the data had to come from an external source, and Keggle doesn't support direct linking to the dataset. Github was used to provide the data as raw CSV

        example_data = 'https://raw.githubusercontent.com/fuzzballb/nd00333-capstone/master/starter_file/cleaned_data.csv'

## Automated ML
First, autoML was configured to optimize for accuracy. and the task is set to do a classification. The column "Exited" is chosen as the label for the training, since that column specifies is a client has left the bank or still has an account. The number of cross validations is five.

For the training it is specified that an experiment should time-out if it takes longer then 30 minutes. Four itarations are allowed to run at simmultaniously


### Results
the results of the different modals used by automated ML are the following. In this case the VotingEnsemble gives the highest accuracy (0.8657). 

![best model](https://github.com/fuzzballb/nd00333-capstone/blob/master/Screenshots/AutoML/Best%20modal.PNG?raw=true "best model")

![widget](https://github.com/fuzzballb/nd00333-capstone/blob/master/Screenshots/AutoML/Widget.PNG?raw=true "widget")

The data provided was balanced, didn't contain missing values and didn't have high cardinality features 

The hyper parameters that where used in this training run where
iterations: 10,
primary_metric:  'accuracy'                     
n_cross_validations: 5

These results can be improved by adding more itterations and cross validations to the AutoML method. 

## Hyperparameter Tuning
The modal that was used to determain if a client would leave the bank, is a LogisticRegression. With Logistic regression the outcome is ether true or false. In our case this is exectly what we are trying to predict.

The strategy used for finding the optimal hyper parameters is RandomParameterSampling. 

The paremeters we that where used was a uniform value between 0.2 and 15 that determaind the "Inverse of regularization strength", whereby smaller values cause stronger regularization. this parameter is added to minimize the chance of overfitting. 

the other paremeter is the maximum number of iterations to converge. Here we specify this to be ether 100 or 300. 


### Results
The best Accuracy for the Hyperparameter tuning was: 0.812, using the folowing parameters

Inverse of regularization strength: 3.5888602469365685
Maximum amount of iterations: 100

For the Hyperparameter tuning, running tests with the other two available strategies (Grid sampling or Bayesian sampling) might also produce more optimal hyperparameters

![best model]https://github.com/fuzzballb/nd00333-capstone/blob/master/Screenshots/HyperParameters/Best%20model.PNG?raw=true "best model")

![widget]https://github.com/fuzzballb/nd00333-capstone/blob/master/Screenshots/HyperParameters/Rundetails.PNG?raw=true "widget")


## Model Deployment
First the best modal is saved as a static file (pkl). Then a new anaconda environment with a AciWebservice is created that runs score.py and has the pip packages specified in conda_env_v_1_0_0.yml. The score.py uses the saved modal to do the predictions. The data that can be sent to this webservice needs to be formatted to the JSON standaard, and the result will return as JSON as well

*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.



## Screen Recording
See the following link (https://www.youtube.com/watch?v=LenZ2_Ed71k) for a screen recording containing 

- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

