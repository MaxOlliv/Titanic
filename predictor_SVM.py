import pandas as pd
import numpy as np
from sklearn import svm
import utilitaires as util
import predictor as p 

def model_SVM(train,test,label):
    SVM = svm.SVC(kernel='rbf')
    
    SVM.fit(train[label],train["Survived"])

    test_predict = pd.DataFrame.copy(test)
    test_predict["Survived"] = SVM.predict(test_predict[label])
    return test_predict

if __name__ == '__main__':
    func_train = util.adaptData
    func_test = util.adaptData
    na = util.getNumAdaptData()
    if na == 1:
        label = np.asarray(["Pclass","Sex","Age","SibSp"])
    elif na == 2:
        label = np.asarray(['Pclass', 'Sex', 'SP', 'ES'])
    elif na == 3:
        label = np.asarray(["Pclass","Sex","Title","Famille"])
    elif na == 4:
        label = np.asarray(['Pclass', 'Sex', 'Parch', 'ES', 'Family', 'Title'])
    elif na == 5:
        label = np.asarray(['Age', 'Sex', 'Fare_bin', 'EC', 'EQ', 'ES', 'Family', 'Title', 'ASP', 'DP'])
    else:
        label = util.getParam()
    model = model_SVM
    path = "../Predictions/SVM.csv"
    
    taux = p.predictor(func_train,func_test,label,model,path)

