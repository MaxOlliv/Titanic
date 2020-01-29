import pandas as pd
from sklearn import discriminant_analysis
import utilitaires as util
import numpy as np
import predictor as p 

def model_lda(train,test,label):
    
    reglin = discriminant_analysis.LinearDiscriminantAnalysis()
    reglin.fit(train[label],train["Survived"])

    test_predict = pd.DataFrame.copy(test)
    test_predict["Survived"] = reglin.predict(test_predict[label])

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
    label = np.asarray(['Pclass', 'Sex', 'Age', 'SP', 'ES'])
elif na == 3:
    label = np.asarray(["Sex","Embarked","Title","Famille"])
else:
    label = util.getParam()
model = model_lda
path = "../Predictions/lda.csv"

p.predictor(func_train,func_test,label,model,path)
