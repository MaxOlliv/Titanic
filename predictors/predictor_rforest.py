import pandas as pd
import numpy as np
import sklearn.ensemble as ske
import utilitaires as util
import predictor as p 

def model_rforest(train,test,label):
    rf = ske.RandomForestClassifier(n_estimators=100,criterion="gini")
    
    rf.fit(train[label],train["Survived"])

    test_predict = pd.DataFrame.copy(test)
    test_predict["Survived"] = rf.predict(test_predict[label])
    return test_predict

func_train = util.adaptData
func_test = util.adaptData
na = util.getNumAdaptData()
if na == 1:
    label = np.asarray(["Pclass","Sex","Age","SibSp"])
elif na == 2:
    label = np.asarray(['Pclass', 'Sex', 'Age', 'Fare', 'EQ'])
elif na == 3:
    label = np.asarray(["Sex","Title","Famille"])
elif na == 4:
    label = np.asarray(['Pclass', 'Sex', 'SibSp', 'EC', 'EQ', 'Title'])
else:
    label = util.getParam()
model = model_rforest
path = "../Predictions/rforest.csv"

p.predictor(func_train,func_test,label,model,path)
