import predictor_knn as knn
import utilitaires as util
import best_formula as bf

func_train = util.adaptData
func_test = util.adaptData
param = util.getParam()
model = knn.model_knn

if __name__ == '__main__':
    bf.best_formula(func_train,func_test,param,model)
