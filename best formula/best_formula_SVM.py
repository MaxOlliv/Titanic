import predictor_SVM as svm 
import utilitaires as util
import best_formula as bf

func_train = util.adaptData
func_test = util.adaptData
param = util.getParam()
model = svm.model_SVM

if __name__ == '__main__':
    tenfold = bf.best_formula(func_train,func_test,param,model)