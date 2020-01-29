import pandas as pd
from sklearn import naive_bayes
import utilitaires as util
import numpy as np
import predictor as p 

def model_naive_bayes(train,test,label):
    
    reglin = naive_bayes.GaussianNB()
    reglin.fit(train[label],train["Survived"])

    test_predict = pd.DataFrame.copy(test)
    test_predict["Survived"] = reglin.predict(test_predict[label])

    test_predict.loc[test_predict.Survived >= 0.5,"Survived"] = 1
    test_predict.loc[test_predict["Survived"] < 0.5,"Survived"] = 0
    test_predict["Survived"] = test_predict["Survived"].astype(int)
    return test_predict

func_train = util.adaptData
func_test = util.adaptData
label = np.asarray(["Sex","SibSp"])
model = model_naive_bayes
path = "../Predictions/naive_bayes.csv"

p.predictor(func_train,func_test,label,model,path)
