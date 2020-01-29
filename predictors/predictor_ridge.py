import pandas as pd
import utilitaires as util
from sklearn import linear_model
import numpy as np
import predictor as p 

def model_ridge(train,test,label):

    r= linear_model.Ridge (alpha = 5)
    r.fit(train[label], train["Survived"]) 

    test_predict = pd.DataFrame.copy(test)
    test_predict["Survived"] = r.predict(test_predict[label])

    test_predict.loc[test_predict.Survived >= 0.5,"Survived"] = 1
    test_predict.loc[test_predict["Survived"] < 0.5,"Survived"] = 0
    test_predict["Survived"] = test_predict["Survived"].astype(int)
    return test_predict
    
func_train = util.adaptData
func_test = util.adaptData
label = np.asarray(['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])
model = model_ridge
path = "../Predictions/ridge.csv"

p.predictor(func_train,func_test,label,model,path)
