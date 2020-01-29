
import pandas as pd
from sklearn import linear_model
import utilitaires as util
import numpy as np
import predictor as p 

def model_ard(train,test,label):
    
    regARD = linear_model.ARDRegression()
    regARD.fit(train[label],train["Survived"])
    
    test_predict = pd.DataFrame.copy(test)
    test_predict["Survived"] = regARD.predict(test_predict[label])
    
    test_predict.loc[test_predict.Survived >= 0.57,"Survived"] = 1
    test_predict.loc[test_predict["Survived"] < 0.57,"Survived"] = 0
    test_predict["Survived"] = test_predict["Survived"].astype(int)

    return test_predict

func_train = util.adaptData
func_test = util.adaptData
label = np.asarray(["Pclass","Sex","Age","SibSp"])
model = model_ard
path = "../Predictions/regression_ard.csv"

p.predictor(func_train,func_test,label,model,path)