#0.796853932584 pour ['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

import predictor_naive_baye as rln
import utilitaires as util
import best_formula as bf

func_train = util.adaptData
func_test = util.adaptData
param = util.getParam()
model = rln.model_naive_bayes

if __name__ == '__main__':
    bf.best_formula(func_train,func_test,param,model)