import pandas as pd
import utilitaires as util
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import predictor as p 

def model_gradboost(train, test, label):

    gradb = GradientBoostingClassifier(learning_rate=0.005, n_estimators=250, max_depth=10, subsample=0.5, max_features=0.5)
    gradb.fit(train[label], train["Survived"]) 

    test_predict = pd.DataFrame.copy(test) 
    test_predict["Survived"] = gradb.predict(test_predict[label])

    return test_predict
    
func_train = util.adaptData
func_test = util.adaptData
na = util.getNumAdaptData()
if na == 1:
    label = np.asarray(['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'])
elif na == 2:
    label = np.asarray(['Pclass', 'Sex', 'Age', 'SibSp', 'SP', 'Fare', 'EQ', 'ES'])
elif na == 3:
    label = np.asarray(["Sex","Title","Famille"])
elif na == 4:
    label = np.asarray(['Age', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Title'])
elif na == 5:
    label = np.asarray(["Age","Pclass","Sex","SibSp","Parch","Fare_bin","EC","EQ","ES","Family","Title","Deck","ASP"])
else:
    label = util.getParam()
model = model_gradboost
path = "../Predictions/gradient_boosting.csv"

p.predictor(func_train,func_test,label,model,path)
