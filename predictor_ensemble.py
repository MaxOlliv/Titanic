import pandas as pd
import numpy as np
import sklearn.ensemble as ske
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn import svm
import sklearn.semi_supervised as sm
import utilitaires as util 
import predictor as p

def model_ensemble(train,test,label):
    cl1 = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100)
    cl2 = KNeighborsClassifier(n_neighbors=3)
    cl3 = sm.LabelPropagation(kernel='rbf', gamma=20, n_neighbors=3, alpha=1, max_iter=100, tol=0.0001)
    cl4 = svm.SVC(kernel='rbf', probability=True)
    cl5 = linear_model.LogisticRegression()
    cl6 = ske.RandomForestClassifier(n_estimators=100,criterion="gini")
    
    rf = ske.VotingClassifier(estimators=[('gradient boost', cl1), ('knn', cl2),
                                          ('labelprop', cl3), ('svm', cl4), ('logistic reg', cl5),
                                            ('rforest', cl6)], voting='soft', weights=[2,4,3,5,2,3])
    cl1.fit(train[label],train["Survived"])
    cl2.fit(train[label],train["Survived"])
    cl3.fit(train[label],train["Survived"])
    cl4.fit(train[label],train["Survived"])
    cl5.fit(train[label],train["Survived"])
    cl6.fit(train[label],train["Survived"])
    rf.fit(train[label],train["Survived"])

    test_predict = pd.DataFrame.copy(test)
    test_predict["Survived"] = rf.predict(test_predict[label])
    return test_predict

func_train = util.adaptData
func_test = util.adaptData
na = util.getNumAdaptData()
if na == 1:
    label = np.asarray(['Pclass', 'Sex', 'Age', 'SibSp', 'Parch'])
elif na == 2:
    label = np.asarray(['Pclass', 'Sex', 'Age', 'Fare'])
elif na == 5:
    label = np.asarray(["Age","Pclass","Sex","SibSp","Parch","Fare_bin","EC","EQ","ES","Family","Title","Deck","ASP"])
else:
    label = util.getParam()
model = model_ensemble
path = "../Predictions/ensemble.csv"

p.predictor(func_test,func_test,label,model,path)
