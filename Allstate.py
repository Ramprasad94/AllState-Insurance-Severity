#Program to run the follwing machine learning models:
#	1.Random forest regression
#	2. Linear regression
#	3. Gradient Boosting regression
import pandas as pd;
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

#read the data in
data = pd.read_csv("train.csv")
#dataset.head(10)
#dataset.index
#dataset.values
#dataset.describe()
#dataset.shape


#drop the first ID column
data = data.drop(data.columns[[0]], axis = 1)


#check for the unique values in each column and determine their count
var = []
for i in range(1,99):
    var.append('cat'+str(i))

#to identify count of unique values in features
list = []
for v in var:
    list.append(len(data[v].unique()))


#list to keep track of categorical variables to perform One Hot Encoding
var_to_encode=[]
var_to_encode = var


#convert categorical features to binary valued discrete features using One Hot Encoding
data = pd.get_dummies(data, columns = var_to_encode)

#copy the class feature and append it to the end
data['class'] = data['loss']
del data['loss']

#drop catergorical features which have unique count of more than 8, in this case from cat99 to cat117
drop_list = []
for i in range(99,117):
    dl = 'cat'+str(i)
    drop_list.append(dl)

for x in drop_list:    
    del data[x]
  
newdata = data


#define a function to calculate rmse metric
def rmse(pred, labels):
    return np.sqrt(np.mean((pred-labels) **2))

######################################################################################    
#Code fragment to run trained model on test data
#read in test data
#testdata = pd.read_csv("test.csv")

#drop the first ID column
#testdata = testdata.drop(testdata.columns[[0]], axis = 1)

#check for the unique values in each column and determine their count
#testvar = []
#for i in range(1,99):
#    testvar.append('cat'+str(i))
#print var
#testlist = []
#for v in testvar:
#    testlist.append(len(data[v].unique()))

#testvar_to_encode=[]
#testvar_to_encode = testvar


#convert categorical features to binary valued discrete features using One Hot Encoding

#testdata = pd.get_dummies(testdata, columns = testvar_to_encode)

#testdrop_list = []
#for i in range(99,117):
#    testdl = 'cat'+str(i)
#    testdrop_list.append(testdl)

#for x in testdrop_list:    
#    del testdata[x]

#newtestdata = testdata
#testarray = newtestdata.values
array = newdata.values
#X_train = array[:,:285]
#Y_train = array[:,285]
#X_test = testarray[:,:285]

###################################################################################


seed = 7    #set seed = 7 to ensure data sample remains the same
X = array[:,:285]
Y = array[:,285]
validation_size = 0.20  #set 20% of training data for tuning set

#split the data into training and tuning set
X_train, X_validation, Y_train, Y_validation= cross_validation.train_test_split(X,Y,test_size=validation_size,random_state=seed)

#####################################################################################

#Random Forest Model
#rf_model = RandomForestRegressor(max_depth = 10, n_estimators = 100, max_features = 'sqrt')
#rf_model.fit(X_train, Y_train)
#Y_pred_rf = rf_model.predict(X_test)
#Y_pred_rf = rf_model.predict(X_validation)
#Y_pred_rf = pd.Series(Y_pred_rf)
#Y_pred_rf.to_csv("final_predictions_rf.csv", index=False)
#print "RMSE for Random Forest regressor: ", rmse(Y_pred_rf, Y_validation)
#print "Mean absolute error: ",mean_absolute_error(Y_validation, Y_pred_rf)

#######################################################################################

#Gradient Boosting Model
#gbm_model = GradientBoostingRegressor(learning_rate = 0.1, n_estimators = 140, max_depth = 10, max_features = 'sqrt')
#gbm_model.fit(X_train,Y_train)    
#y_pred_gbm = gbm_model.predict(X_validation)
#y_pred_gbm = gbm_model.predict(X_test)
#y_pred_gbm = pd.Series(y_pred_gbm)
#y_pred_gbm.to_csv("final_predictions_gbm_best.csv", index=False)
#print "Root Mean Squared Error (RMSE) for Gradient Boosting Regressor: ", rmse(y_pred_gbm, Y_validation)
#print "Mean Absolute Error(MAE) for Gradient Boosting Regressor : ",mean_absolute_error(Y_validation, y_pred_gbm)

#######################################################################################

#SVM Model code
from sklearn.svm import SVR
svm = SVR(C = 1.0, epsilon = 0.2)
svm.fit(X_train, Y_train)
Y_pred_svm = svm.predict(X_validation)    
print "RMSE for SVM REGRESSOR: ", rmse(Y_pred_svm, Y_validation)
print "Mean absolute error: ",mean_absolute_error(Y_validation, Y_pred_svm)

######################################################################################    
  
#Linear Regression Model
#lrmodel = LinearRegression()
#lrmodel.fit(X_train, Y_train)
#Y_pred_lr = lrmodel.predict(X_validation)
#predictedY = pd.Series(Y_pred_lr)
#predictedY.to_csv("lr_predicted.csv")
#print "RMSE for linear regression: ", rmse(Y_pred_lr, Y_validation)
#print "Mean absolute error: ",mean_absolute_error(Y_validation, Y_pred_lr)




   
    
