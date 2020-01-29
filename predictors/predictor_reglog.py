import pandas as pd
from sklearn import linear_model
import utilitaires as util
import numpy as np
import predictor as p 

def model_reglog(train,test,label):
    
    reglog = linear_model.LogisticRegression()
    reglog.fit(train[label],train["Survived"])
    
    test_predict = pd.DataFrame.copy(test)
    test_predict["Survived"] = reglog.predict(test_predict[label])

    return test_predict

func_train = util.adaptData
func_test = util.adaptData
na = util.getNumAdaptData()
label = []
if na == 1:
    label = np.asarray(["Pclass","Sex","Age","SibSp"])
elif na == 2:
    label = np.asarray(['Pclass', 'Sex', 'Age', 'SP', 'EQ'])
elif na == 5:
    label = np.asarray(['ES', 'Family', 'Title', 'ASP'])
else:
    label = util.getParam()
model = model_reglog
path = "../Predictions/regression_logistique.csv"

p.predictor(func_train,func_test,label,model,path)