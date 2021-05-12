#exec(open('format_data.py').read())
import subprocess as sp
import pandas as pd
import pickle as pk
from sklearn.datasets import load_boston
from sklearn import preprocessing
import numpy as np
import importlib as il

if __name__ == "__main__":
    sp.call('cls', shell = True)

    # initialize data configuration dictionary
    datacfg = dict()

    # iris dataset
    dfIris = pd.read_csv('.\\data\\iris\\iris.data'
        ,names = [
             'sepal_length'
            ,'sepal_width'
            ,'petal_length'
            ,'petal_width'
            ,'class'
        ]
        ,header = None
        ,index_col = False)
    dfIris.loc[34, 'petal_width'] = 0.2
    dfIris.loc[37, 'sepal_width'] = 3.6
    dfIris.loc[37, 'petal_length'] = 1.4
    pk.dump(dfIris, open('.\\data\\iris.pkl','wb'))

    # boston housing dataset
    ds = load_boston()
    X = ds.data
    y = ds.target
    features = [feature for feature in ds['feature_names']]
    targets = ['PRICE']
    data = np.concatenate([X,y[:,np.newaxis]], axis = 1)
    dfBoston = pd.DataFrame(data = data, columns = features + targets)
    pk.dump(dfBoston, open('.\\data\\bostonHousing.pkl','wb'))

    # wine quality dataset
    dfWineQualityRed = pd.read_csv('.\\data\\wineQuality\\winequality-red.csv'
        ,header = 0)
    pk.dump(dfWineQualityRed, open('.\\data\\winequality-red.pkl','wb'))
    dfWineQualityWhite = pd.read_csv('.\\data\\wineQuality\\winequality-white.csv'
        ,header = 0)
    pk.dump(dfWineQualityWhite, open('.\\data\\winequality-white.pkl','wb'))

    # abalone dataset
    dfAbalone = pd.read_csv('.\\data\\abalone\\abalone.data'
        ,names = ['sex'
            ,'length'
            ,'diameter'
            ,'height'
            ,'whole weight'
            ,'shucked weight'
            ,'viscera weight'
            ,'shell weight'
            ,'rings'])
    pk.dump(dfAbalone, open('.\\data\\abalone.pkl','wb'))

    # auto mpg dataset
    dfAutoMpg = pd.read_csv('.\\data\\autoMpg\\auto-mpg.data'
        ,names = ['mpg'
            ,'cylinders'
            ,'displacement'
            ,'horsepower'
            ,'weight'
            ,'acceleration'
            ,'model year'
            ,'origin'
            ,'car name'])
    dfAutoMpg.drop(['car name'], axis = 1, inplace = True)
    idx = np.logical_not(dfAutoMpg.loc[:,'horsepower'] == '?')
    dfAutoMpg = dfAutoMpg.loc[idx,:]
    dfAutoMpg.loc[:,'horsepower'] = dfAutoMpg.loc[:,'horsepower'].astype(np.float64)
    pk.dump(dfAutoMpg, open('.\\data\\autoMpg.pkl','wb'))

    # bank marketing dataset
    dfBankMarketing = pd.read_csv('.\\data\\bankMarketing\\bank.csv'
        ,header = 0)
    pk.dump(dfBankMarketing, open('.\\data\\bankMarketing.pkl','wb'))

    # heart disease wisconsin data set
    dfHeartDisease = pd.read_csv('.\\data\\heartDiseaseWisconsin\\processed.cleveland.data'
        ,names = ['age'
            ,'sex'
            ,'cp'
            ,'trestbps'
            ,'chol'
            ,'fbs'
            ,'restecg'
            ,'thalach'
            ,'exang'
            ,'oldpeak'
            ,'slope'
            ,'ca'
            ,'thal'
            ,'num'])
    pk.dump(dfHeartDisease, open('.\\data\\heartDiseaseWisconsin.pkl','wb'))

    # wine cultivars data set
    dfWineCult = pd.read_csv('.\\data\\wineCultivar\\wine.data'
        ,names = ['cultivar'
            ,'alcohol'
            ,'malic acid'
            ,'ash'
            ,'alcalinity of ash  '
            ,'magnesium'
            ,'total phenols'
            ,'flavanoids'
            ,'nonflavanoid phenols'
            ,'proanthocyanins'
            ,'color intensity'
            ,'hue'
            ,'od280/od315 of diluted wines'
            ,'proline'])
    pk.dump(dfWineCult, open('.\\data\\wineCultivar.pkl','wb'))

    # pima indians diabetes dataset
    dfPima = pd.read_csv('.\\data\\pima\\pima-indians-diabetes.csv'
        ,names = ['pregnancies'
            ,'glucose'
            ,'bloodPressure'
            ,'skinThickness'
            ,'insulin'
            ,'bmi'
            ,'diabetesPedigreef'
            ,'age'
            ,'class'])
    pk.dump(dfPima, open('.\\data\\pima.pkl','wb'))

    # adult census income dataset
    dfAdult = pd.read_csv('.\\data\\adult\\adult_all.data'
        ,names = ['age'
            ,'workclass'
            ,'fnlwgt'
            ,'education'
            ,'education-num'
            ,'marital-status'
            ,'occupation'
            ,'relationship'
            ,'race'
            ,'sex'
            ,'capital-gain'
            ,'capital-loss'
            ,'hours-per-week'
            ,'native-country'
            ,'class'])
    pk.dump(dfAdult, open('.\\data\\adult.pkl','wb'))