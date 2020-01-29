#0.796866416979 pour ('Pclass', 'Sex', 'Age', 'Parch',  'Embarked')

import predictor_shrinkage as rln
import utilitaires as util
import best_formula as bf

func_train = util.adaptData
func_test = util.adaptData
param = util.getParam()
model = rln.model_shrinkage

if __name__ == '__main__':
    bf.best_formula(func_train,func_test,param,model)