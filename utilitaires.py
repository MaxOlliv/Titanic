import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import KFold
import string
from sklearn import linear_model


#Validation 10 fold
def tenfold(data,label,model):

    kf = KFold(len(data),n_folds=10)
    res = 0.
    for train,test in kf:
        train_actual = data.ix[[data.index[i] for i in train],0::]
        test_actual = data.ix[[data.index[i] for i in test],0::]
        test_predict = model(train_actual,test_actual,label)
        res_temp = 0.
        for i in [data.index[i] for i in test]:
            if test_actual["Survived"][i] == test_predict["Survived"][i]:
                res_temp+=1.
        res+=res_temp/len(test)

    return res/10

#Gestion des variables qualitatives et suppression des lignes non completes
def reducedData(data):
    data = data.drop(["Ticket","Cabin"], axis=1) 
    data = data.dropna()
    label_encoder = preprocessing.LabelEncoder()
    data["Sex"] = label_encoder.fit_transform(data["Sex"])
    data["Embarked"] = label_encoder.fit_transform(data["Embarked"])
    return data

#Gestion des variables qualitatives et des NaN

numAdaptData = 4

param1 = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
param2 = ["Pclass","Sex","Age","SibSp","Parch","SP","Fare","EC","EQ","ES"]
param3 = ["Pclass","Sex","Age","Fare","Embarked","Title","Famille"]
param4 = ["Age","Pclass","Sex","SibSp","Parch","Fare","EC","EQ","ES","Family","Title","Deck"]
param5 = ["Age","Pclass","Sex","Fare_bin","EC","EQ","ES","Family","Title","Deck","ASP","DP"]

def getNumAdaptData():
    return numAdaptData

def getParam():
    if numAdaptData == 1:
        return param1
    if numAdaptData == 2:
        return param2
    if numAdaptData == 3:
        return param3
    if numAdaptData == 4:
        return param4
    if numAdaptData == 5:
        return param5
    else:
        print "numAdaptData a corriger dans utilitaires"
    return []

def adaptData(data):
    if numAdaptData == 1:
        return adaptData1(data)
    if numAdaptData == 2:
        return adaptData2(data)
    if numAdaptData == 3:
        return adaptData3(data)
    if numAdaptData == 4:
        return adaptData4(data)
    if numAdaptData == 5:
        return adaptData5(data)
    else:
        return reducedData(data)

#Plusieurs possibilites d'adaptation des donnees
def adaptData1(data):
    data = data.drop(["Ticket","Cabin","Name"], axis=1) 
    label_encoder = preprocessing.LabelEncoder()
    data["Sex"] = label_encoder.fit_transform(data["Sex"])
    median_ages = np.zeros((2,3))
    for i in range(0, 2):
        for j in range(0, 3):
            median_ages[i,j] = data[(data["Sex"] == i) & (data["Pclass"] == j+1)]["Age"].dropna().median()
    for i in range(0, 2):
        for j in range(0, 3):
            data.loc[ (data.Age.isnull()) & (data.Sex == i) & (data.Pclass == j+1),"Age"] = median_ages[i,j]
            
    median_fare = np.zeros((2,3))
    for i in range(0, 2):
        for j in range(0, 3):
            median_fare[i,j] = data[(data["Sex"] == i) & (data["Pclass"] == j+1)]["Fare"].dropna().median()
    for i in range(0, 2):
        for j in range(0, 3):
            data.loc[ (data.Fare.isnull()) & (data.Sex == i) & (data.Pclass == j+1),"Fare"] = median_fare[i,j]

    data["Embarked"] = data["Embarked"].fillna("S")
    data["Embarked"] = label_encoder.fit_transform(data["Embarked"])
    data["Fare"] = data["Fare"].fillna(data["Fare"].mean())
    
    return data

def adaptData2(data):
    
    data = data.drop(["Ticket","Cabin","Name"], axis=1) 
    
    label_encoder = preprocessing.LabelEncoder()
    data["Sex"] = label_encoder.fit_transform(data["Sex"])
    
    data.loc[data["Age"].isnull(),"Age"] = data["Age"].dropna().mean()
    data.loc[data["Fare"].isnull(),"Fare"] = data["Fare"].dropna().mean()
    
    data["SP"] = 0
    data["SP"] = data["SibSp"]+data["Parch"]
    #data = data.drop(["SibSp","Parch"], axis=1) 
    
    data["EC"] = 0
    data["EQ"] = 0
    data["ES"] = 0
    data["Embarked"] = data["Embarked"].fillna("S")
    data.loc[data["Embarked"]=="C","EC"] = 1
    data.loc[data["Embarked"]!="C","EC"] = 0
    data.loc[data["Embarked"]=="Q","EQ"] = 1
    data.loc[data["Embarked"]!="Q","EQ"] = 0
    data.loc[data["Embarked"]=="S","ES"] = 1
    data.loc[data["Embarked"]!="S","ES"] = 0
    data = data.drop(["Embarked"], axis=1) 
    
    data["Pclass"] = normalize(data["Pclass"])
    data["Age"] = normalize(data["Age"])
    data["SP"] = normalize(data["SP"])
    data["Fare"] = normalize(data["Fare"])
    
    return data

def adaptData3(data):
    data = data.drop(["Ticket","Cabin"], axis=1)
    label_encoder = preprocessing.LabelEncoder()
    data["Sex"] = label_encoder.fit_transform(data["Sex"])
    median_ages = np.zeros((2,3))
    for i in range(0, 2):
        for j in range(0, 3):
            median_ages[i,j] = data[(data["Sex"] == i) & (data["Pclass"] == j+1)]["Age"].dropna().median()
    for i in range(0, 2):
        for j in range(0, 3):
            data.loc[ (data.Age.isnull()) & (data.Sex == i) & (data.Pclass == j+1),"Age"] = median_ages[i,j]
            
    median_fare = np.zeros((2,3))
    for i in range(0, 2):
        for j in range(0, 3):
            median_fare[i,j] = data[(data["Sex"] == i) & (data["Pclass"] == j+1)]["Fare"].dropna().median()
    for i in range(0, 2):
        for j in range(0, 3):
            data.loc[ (data.Fare.isnull()) & (data.Sex == i) & (data.Pclass == j+1),"Fare"] = median_fare[i,j]

    data["Embarked"] = data["Embarked"].fillna("S")
    data["Embarked"] = label_encoder.fit_transform(data["Embarked"])
    data["Fare"] = data["Fare"].fillna(data["Fare"].mean())

    data["Title"] = map(titre,data["Name"])
    data.loc[(data["Title"] == 'Mlle'),"Title"] = 'Miss'
    data.loc[(data["Title"] == 'Ms'),"Title"] = 'Miss'
    data.loc[(data["Title"] == 'Mme'),"Title"] = 'Mrs'
    data.loc[(data["Title"] == 'Dona') | (data["Title"] ==  'Lady') | (data["Title"] == 'the Countness') |(data["Title"] == 'Capt') |(data["Title"] == 'Col') | (data["Title"] == 'Don') | (data["Title"] == 'Dr') | (data["Title"] == 'Major') | (data["Title"] == 'Rev') | (data["Title"] == 'Sir') |(data["Title"] == 'Jonkheer'),"Title"]  = 'Rare'

    
    data["Nbre"] = data["SibSp"] + data["Parch"] +1
    data["Famille"] = data["Nbre"]
    data.loc[(data["Nbre"] == 1), "Famille"] = 'seul'
    data.loc[(data["Nbre"] > 1) & (data["Nbre"]<= 4),"Famille"] = 'petite'
    data.loc[(data["Nbre"] > 4),"Famille"] = 'grande'
    
    data = data.drop(["Name","Parch","SibSp","Nbre"], axis=1) 
    data["Title"] = label_encoder.fit_transform(data["Title"])
    data["Famille"] = label_encoder.fit_transform(data["Famille"])
    return data
    
def adaptData4(data):
    
    label_encoder = preprocessing.LabelEncoder()
    
    data["Sex"] = label_encoder.fit_transform(data["Sex"])
    
    data.loc[data["Fare"].isnull(),"Fare"] = data["Fare"].dropna().mean()
    
    data["EC"] = 0
    data["EQ"] = 0
    data["ES"] = 0
    data["Embarked"] = data["Embarked"].fillna("S")
    data.loc[data["Embarked"]=="C","EC"] = 1
    data.loc[data["Embarked"]!="C","EC"] = 0
    data.loc[data["Embarked"]=="Q","EQ"] = 1
    data.loc[data["Embarked"]!="Q","EQ"] = 0
    data.loc[data["Embarked"]=="S","ES"] = 1
    data.loc[data["Embarked"]!="S","ES"] = 0
    
    data["Family"] = 0
    data["Family"] = data["SibSp"]+data["Parch"]
    
    title_list=['Mrs','Mr','Master','Miss','Major','Rev','Dr','Ms','Mlle','Col','Capt','Mme','Countess','Don','Jonkheer']
    data['Title'] = data['Name'].map(lambda x: substrings_in_string(x, title_list))
    data['Title'] = data.apply(replace_titles,axis=1)
    data["Title"] = label_encoder.fit_transform(data["Title"])
    
    cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
    data.loc[data["Cabin"].isnull(),"Cabin"] = "Unknown"
    data['Deck'] = data['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))
    data["Deck"] = label_encoder.fit_transform(data["Deck"])
    
    data = setMissingAges(data,param4)
    
    scaler = preprocessing.StandardScaler()
    data["Age"] = scaler.fit_transform(data["Age"].reshape(-1,1))
    data["Pclass"] = scaler.fit_transform(data["Pclass"].astype(float).reshape(-1,1))
    data["Sex"] = scaler.fit_transform(data["Sex"].astype(float).reshape(-1,1))
    data["SibSp"] = scaler.fit_transform(data["SibSp"].astype(float).reshape(-1,1))
    data["Parch"] = scaler.fit_transform(data["Parch"].astype(float).reshape(-1,1))
    data["Fare"] = scaler.fit_transform(data["Fare"].reshape(-1,1))
    data["EC"] = scaler.fit_transform(data["EC"].astype(float).reshape(-1,1))
    data["EQ"] = scaler.fit_transform(data["EQ"].astype(float).reshape(-1,1))
    data["ES"] = scaler.fit_transform(data["ES"].astype(float).reshape(-1,1))
    data["Family"] = scaler.fit_transform(data["Family"].astype(float).reshape(-1,1))
    data["Title"] = scaler.fit_transform(data["Title"].astype(float).reshape(-1,1))
    data["Deck"] = scaler.fit_transform(data["Deck"].astype(float).reshape(-1,1))
    
    data = data.drop(["Ticket","Cabin","Name","Embarked"], axis=1)
    
    return data


def adaptData5(data):
    
    label_encoder = preprocessing.LabelEncoder()
    
    data["Sex"] = label_encoder.fit_transform(data["Sex"])
    
    data.loc[data["Fare"].isnull(),"Fare"] = data["Fare"].dropna().mean()
    
    data["EC"] = 0
    data["EQ"] = 0
    data["ES"] = 0
    data["Embarked"] = data["Embarked"].fillna("S")
    data.loc[data["Embarked"]=="C","EC"] = 1
    data.loc[data["Embarked"]!="C","EC"] = 0
    data.loc[data["Embarked"]=="Q","EQ"] = 1
    data.loc[data["Embarked"]!="Q","EQ"] = 0
    data.loc[data["Embarked"]=="S","ES"] = 1
    data.loc[data["Embarked"]!="S","ES"] = 0
    
    data["Family"] = 0
    data["Family"] = data["SibSp"]+data["Parch"]
    
    title_list=['Mrs','Mr','Master','Miss','Major','Rev','Dr','Ms','Mlle','Col','Capt','Mme','Countess','Don','Jonkheer']
    data['Title'] = data['Name'].map(lambda x: substrings_in_string(x, title_list))
    data['Title'] = data.apply(replace_titles,axis=1)
    data["Title"] = label_encoder.fit_transform(data["Title"])
    
    cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
    data.loc[data["Cabin"].isnull(),"Cabin"] = "Unknown"
    data['Deck'] = data['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))
    data["Deck"] = label_encoder.fit_transform(data["Deck"])
    
    data = binning(data,'Fare','Fare_bin',4)
    
    param_reglin = ["Age","Pclass","Sex","SibSp","Parch","Fare_bin","EC","EQ","ES","Family","Title","Deck"]
    data = setMissingAges(data,param_reglin)
    
    data["ASP"] = 0
    data["ASP"] = data["Age"]*data["Sex"]*data["Pclass"]
    
    data["DP"] = 0
    data["DP"] = data["Deck"]*data["Pclass"]
    
    scaler = preprocessing.StandardScaler()
    data["Age"] = scaler.fit_transform(data["Age"].reshape(-1,1))
    data["Pclass"] = scaler.fit_transform(data["Pclass"].astype(float).reshape(-1,1))
    data["Sex"] = scaler.fit_transform(data["Sex"].astype(float).reshape(-1,1))
    data["SibSp"] = scaler.fit_transform(data["SibSp"].astype(float).reshape(-1,1))
    data["Parch"] = scaler.fit_transform(data["Parch"].astype(float).reshape(-1,1))
    data["EC"] = scaler.fit_transform(data["EC"].astype(float).reshape(-1,1))
    data["EQ"] = scaler.fit_transform(data["EQ"].astype(float).reshape(-1,1))
    data["ES"] = scaler.fit_transform(data["ES"].astype(float).reshape(-1,1))
    data["Family"] = scaler.fit_transform(data["Family"].astype(float).reshape(-1,1))
    data["Title"] = scaler.fit_transform(data["Title"].astype(float).reshape(-1,1))
    data["Deck"] = scaler.fit_transform(data["Deck"].astype(float).reshape(-1,1))
    data["Fare_bin"] = scaler.fit_transform(data["Fare_bin"].astype(float).reshape(-1,1))
    data["ASP"] = scaler.fit_transform(data["ASP"].astype(float).reshape(-1,1))
    data["DP"] = scaler.fit_transform(data["DP"].astype(float).reshape(-1,1))
    
    
    data = data.drop(["Ticket","Cabin","Name","Fare","Embarked"], axis=1)
    
    return data


#fonctions auxiliaires 
    
def normalize(column):
    
    x = column.values.astype(float).reshape(-1,1)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    column_normalized = pd.DataFrame(x_scaled)
    
    return column_normalized

def titre():
    temp = str.split(",")[1]
    temp = temp.split(".")[0]
    temp = temp.replace(" ","")
    return temp

def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if string.find(big_string, substring) != -1:
            return substring
    return np.nan

def replace_titles(x):
    title=x['Title']
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Countess', 'Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title =='Dr':
        if x['Sex']=='Male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title
    
def setMissingAges(data,param):
    
    age_data = data[param]
    
    train = age_data.loc[(data.Age.notnull())]
    test = age_data.loc[(data.Age.isnull())]
    
    y = train.values[:, 0]
    x = train.values[:, 1::]
    
    reglin = linear_model.LinearRegression()
    reglin.fit(x, y)
    
    predictedAges = reglin.predict(test.values[:, 1::])
    
    data.loc[data.Age.isnull(),'Age'] = predictedAges 
    
    return data

def binning(data,name,name_bin,div):
    data['temp'] = pd.qcut(data[name], div)
    bin_list = data['temp'].unique()
    data[name_bin] = 0
    for i in range(0,len(bin_list)):
        data.loc[data["temp"]==bin_list[i],name_bin] = i
    data = data.drop(['temp'], axis=1)
    
    return data