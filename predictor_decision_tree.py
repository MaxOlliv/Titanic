import pandas as pd
import utilitaires as util
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import predictor as p 

def model_decTree(train, test, label):

    dectree = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, max_features=None)
    dectree.fit(train[label], train["Survived"]) 

    test_predict = pd.DataFrame.copy(test)
    test_predict["Survived"] = dectree.predict(test_predict[label])

    return test_predict
    
func_train = util.adaptData
func_test = util.adaptData
na = util.getNumAdaptData()
if na == 1:
    label = np.asarray(['Pclass', 'Sex', 'Age'])
elif na == 2:
    label = np.asarray(['Pclass', 'Sex', 'Age'])
elif na == 3:
    label = np.asarray(["Sex","Embarked","Title","Famille"])
elif na == 4:
    label = np.asarray(['Pclass', 'Sex', 'SibSp', 'EQ', 'ES', 'Title'])
else:
    label = util.getParam()
model = model_decTree
path = "../Predictions/decision_tree.csv"

p.predictor(func_train,func_test,label,model,path)
