#0.786729088639pour ['Sex']

import predictor_bernoulli_baye as rln
import utilitaires as util
import best_formula as bf

func_train = util.adaptData
func_test = util.adaptData
param = util.getParam()
model = rln.model_bernoulli_bayes

if __name__ == '__main__':
    bf.best_formula(func_train,func_test,param,model)