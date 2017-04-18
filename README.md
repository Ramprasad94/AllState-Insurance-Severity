# AllState-Insurance-Severity
Repository containing source code for machine learning project to predict severity of insurance claims for AllState Corporation


Steps to run the project:

1. There are four python files in the current directory: 

AllStateGB.py - The main Python file containing our final implemented and tuned model - Gradient Boosting Regression 
AllStateLR.py -  The python file containing the other model we tried out on the dataset
AllStateRF.py - The python file containing the other model we tried out on the dataset
AllStateSVM.py - The python file containing the other model we tried out on the dataset
along with the dataset: train.csv and test.csv

NOTE: AllStateGB is our MAIN python file which contains the model that we tuned and determined the best final predictions.


2. train.csv should be used for training and test.csv should be used for testing. 

3. However, since the test.csv does not contain the class target variable column(this is an ongoing Kaggle competition dataset), we use a validation set from our train.csv
to evaluate our model using RMSE and MAE scores.

4. Command to run on tuning set: 

     python <filename>.py <path to train.csv file> Tuning
     
     Ex: python AllStateGB.py train.csv Tuning
     
5. Command to run on test set:

    python <filename>.py <path to train.csv file> Test <path to test.csv file>

    Ex:  python AllStateGB.py train.csv Test test.csv

