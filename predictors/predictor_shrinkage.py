import pandas as pd
from sklearn import discriminant_analysis
import utilitaires as util
import numpy as np
import predictor as p 

def model_shrinkage(train,test,label):
    
    reglin = discriminant_analysis.LinearDiscriminantAnalysis(shrinkage=0.000001, solver='lsqr')
    reglin.fit(train[label],train["Survived"])

    test_predict = pd.DataFrame.copy(test)
    test_predict["Survived"] = reglin.predict(test_predict[label])

    test_predict.loc[test_predict.Survived >= 0.5,"Survived"] = 1
    test_predict.loc[test_predict["Survived"] < 0.5,"Survived"] = 0
    test_predict["Survived"] = test_predict["Survived"].astype(int)
    return test_predict

func_train = util.adaptData
func_test = util.adaptData
label = np.asarray(['Pclass', 'Sex', 'Age',  'Parch', 'Fare', 'Embarked'])
model = model_shrinkage
path = "../Predictions/shrinkage.csv"

p.predictor(func_train,func_test,label,model,path)
