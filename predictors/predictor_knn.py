import pandas as pd
import utilitaires as util
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import predictor as p 

def model_knn(train,test,label):

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(train[label], train["Survived"]) 

    test_predict = pd.DataFrame.copy(test)
    test_predict["Survived"] = neigh.predict(test_predict[label])

    return test_predict
    
func_train = util.adaptData
func_test = util.adaptData
na = util.getNumAdaptData()
if na == 1:
    label = np.asarray(["Pclass","Sex","Age","SibSp"])
elif na == 2:
    label = np.asarray(['Pclass', 'Sex', 'Age', 'Fare'])
elif na == 3:
    label = np.asarray(["Sex","Embarked","Title","Famille"])
elif na == 4:
    label = np.asarray(['Pclass', 'Sex', 'SibSp', 'EC', 'ES', 'Family', 'Title'])
elif na == 5:
    label = np.asarray(['Age', 'Pclass', 'Sex', 'Fare_bin', 'EQ', 'Family', 'Title', 'Deck'])
else:
    label = util.getParam()
model = model_knn
path = "../Predictions/k_nearest_neighbors.csv"

p.predictor(func_train,func_test,label,model,path)
