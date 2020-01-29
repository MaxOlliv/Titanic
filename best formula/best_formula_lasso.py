#0.79911360799 pour ('Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked')
import predictor_lasso as rln
import utilitaires as util
import best_formula as bf

func_train = util.adaptData
func_test = util.adaptData
param = util.getParam()
model = rln.model_lasso

if __name__ == '__main__':
    bf.best_formula(func_train,func_test,param,model)