import pandas as pd
import utilitaires as util
from sklearn import linear_model
import numpy as np
import predictor as p 

def model_lasso(train,test,label):

    r= linear_model.Lasso()
    r.fit(train[label], train["Survived"]) 

    test_predict = pd.DataFrame.copy(test)
    test_predict["Survived"] = r.predict(test_predict[label])

    test_predict.loc[test_predict.Survived >= 0.5,"Survived"] = 1
    test_predict.loc[test_predict["Survived"] < 0.5,"Survived"] = 0
    test_predict["Survived"] = test_predict["Survived"].astype(int)
    return test_predict
    
func_train = util.adaptData
func_test = util.adaptData
na = util.getNumAdaptData()
if na == 1:
    label = np.asarray(['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])
elif na == 2:
    label = np.asarray(['Pclass', 'Sex', 'SP', 'ES'])
elif na == 3:
    label = np.asarray(["Pclass","Sex","Title","Famille"])
elif na ==4:
    label = np.asarray(['Pclass', 'Sex', 'Parch', 'ES', 'Family', 'Title'])
else:
    label = util.getParam()
model = model_lasso
path = "../Predictions/lasso.csv"

p.predictor(func_train,func_test,label,model,path)
