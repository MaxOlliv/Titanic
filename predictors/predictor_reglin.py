import pandas as pd
from sklearn import linear_model
import utilitaires as util
import numpy as np
import predictor as p 

def model_reglin(train,test,label):
    
    reglin = linear_model.LinearRegression()
    reglin.fit(train[label],train["Survived"])

    test_predict = pd.DataFrame.copy(test)
    test_predict["Survived"] = reglin.predict(test_predict[label])

    test_predict.loc[test_predict.Survived >= 0.5,"Survived"] = 1
    test_predict.loc[test_predict["Survived"] < 0.5,"Survived"] = 0
    test_predict["Survived"] = test_predict["Survived"].astype(int)
    return test_predict

func_train = util.adaptData
func_test = util.adaptData
na = util.getNumAdaptData()
if na == 1:
    label = np.asarray(["Pclass","Sex","Age","SibSp"])
elif na == 2:
    label = np.asarray(['Pclass', 'Sex', 'Age', 'SP', 'ES'])
elif na == 3:
    label = np.asarray(["Sex","Title","Famille"])
else:
    label = util.getParam()
model = model_reglin
path = "../Predictions/regression_lineaire.csv"

p.predictor(func_train,func_test,label,model,path)
