import predictor_rforest as rf
import utilitaires as util
import best_formula as bf

func_train = util.adaptData
func_test = util.adaptData
param = util.getParam()
model = rf.model_rforest

if __name__ == '__main__':
    bf.best_formula(func_train,func_test,param,model)