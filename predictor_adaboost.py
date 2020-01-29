import pandas as pd
import utilitaires as util
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import predictor as p 

def model_adaboost(train,test,label):

    adab = AdaBoostClassifier(learning_rate=0.1, n_estimators=100)
    adab.fit(train[label], train["Survived"]) 

    test_predict = pd.DataFrame.copy(test)
    test_predict["Survived"] = adab.predict(test_predict[label])

    return test_predict
    
func_train = util.adaptData
func_test = util.adaptData
na = util.getNumAdaptData()
if na == 1:
    label = np.asarray(['Pclass', 'Sex', 'Age', 'SibSp', 'Parch'])
elif na == 5:
    label = np.asarray(['Pclass', 'Sex', 'Family', 'Title'])
model = model_adaboost
path = "../Predictions/adaboost.csv"

p.predictor(func_train,func_test,label,model,path)
