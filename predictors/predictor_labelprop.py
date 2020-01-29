import pandas as pd
import numpy as np
import sklearn.semi_supervised as sm
import utilitaires as util 
import predictor as p 

def model_labelprop(train,test,label):
    lp = sm.LabelPropagation(kernel='rbf', gamma=20, n_neighbors=3, alpha=1, max_iter=100, tol=0.0001)
    
    lp.fit(train[label],train["Survived"])

    test_predict = pd.DataFrame.copy(test)
    test_predict["Survived"] = lp.predict(test_predict[label])
    return test_predict

func_train = util.adaptData
func_test = util.adaptData
na = util.getNumAdaptData()
if na == 1:
    label = np.asarray(["Pclass","Sex","Age","SibSp"])
elif na == 2:
    label = np.asarray(['Pclass', 'Sex', 'Age', 'SibSp', 'Parch'])
elif na == 3:
    label = np.asarray(["Sex","Title","Famille"])
elif na == 4:
    label = np.asarray(['Pclass', 'Sex', 'SibSp', 'Fare', 'EC', 'EQ', 'Title'])
elif na == 5:
    label = np.asarray(['Age', 'Sex', 'Fare_bin', 'ES', 'Deck', 'DP'])
else:
    label = util.getParam()
model = model_labelprop
path = "../Predictions/labelprop.csv"

p.predictor(func_train,func_test,label,model,path)