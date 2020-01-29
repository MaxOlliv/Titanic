# ('Pclass', 'Sex', 'Age', 'SibSp', 'Parch')
# 0.810337078652

import predictor_adaboost as ab
import utilitaires as util
import best_formula as bf

func_train = util.adaptData
func_test = util.adaptData
param = util.getParam()
model = ab.model_adaboost

if __name__ == '__main__':
    bf.best_formula(func_train,func_test,param,model)