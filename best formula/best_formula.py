import pandas as pd
import utilitaires as util
import numpy as np
import itertools
from joblib import Parallel, delayed


def iteration(i,loop,subset,train,model):
            
    label = np.asarray(subset)
    temp = util.tenfold(train,label,model)
        
    return temp

#ARGUMENTS :
#func_train : fonction a appliquer aux donnees d'apprentissage
#func_test : fonction a appliquer aux donnees de test
#param : ensemble des variables pouvant servir de label
#model : fonction modele utilise pour la prediction

def best_formula(func_train,func_test,param,model):
    
    train = pd.read_csv('../Data/train.csv')
    test = pd.read_csv('../Data/test.csv')
    
    train = func_train(train)
    test = func_test(test)

    
    loop = 2**len(param)-1;
    subset_table = range(0,loop)
    
    current=0;
    print(str(loop) + " taches a executer.")
    for i in range(1, len(param)+1):
        for subset in itertools.combinations(param, i):
            subset_table[current] = subset
            current+=1
    
    print ""
    tenfold = Parallel(n_jobs=-1,verbose=50)(delayed(iteration)(i,loop,subset,train,model) for subset in subset_table)
       
    argmax = np.argmax(tenfold)       
    print("")
    print("Resultat :")
    print "Meilleur taux obtenu : {0}".format(tenfold[argmax])
    print("pour les parametres suivants :")
    print (subset_table[argmax])
    
    return tenfold