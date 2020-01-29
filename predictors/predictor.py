import pandas as pd
import utilitaires as util 

#ARGUMENTS :
#func_train : fonction a appliquer aux donnees d'apprentissage
#func_test : fonction a appliquer aux donnees de test
#label : variables de prediction
#model : fonction modele utilise pour la prediction
#path : chemin pour l'exportation en csv

def predictor(func_train,func_test,label,model,path):
    
    train = pd.read_csv('../Data/train.csv', header=0)
    test = pd.read_csv('../Data/test.csv', header=0)

    train = func_train(train)
    test = func_test(test)

    taux = util.tenfold(train,label,model)
    print "Taux de predictions correctes pour les parametres actuels : {0}".format(taux)
    
    test = model(train,test,label)
    
    test[["PassengerId","Survived"]].to_csv(path, index=False)
    
    return taux