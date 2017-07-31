import csv
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing

data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

data_train = data_train.drop(['Id','GarageArea','GarageYrBlt','OverallCond','BsmtFinSF2'
                    ,'LowQualFinSF','BsmtHalfBath','KitchenAbvGr','EnclosedPorch','YrSold','LotFrontage'], axis = 1)
data_test = data_test.drop(['Id','GarageArea','GarageYrBlt','OverallCond','BsmtFinSF2'
                    ,'LowQualFinSF','BsmtHalfBath','KitchenAbvGr','EnclosedPorch','YrSold','LotFrontage'], axis = 1)

#Categorizing MSSubClass
def cat_subclass(mssub):
    temp = int(mssub)
    if temp == 60:
        return 2
    elif temp == 30 or temp == 45 or temp == 180:
        return 0
    else:
        return 1

data_train['MSSubClass'] = data_train[['MSSubClass']].apply(cat_subclass, axis=1)
data_train['MSSubClass'] = data_train['MSSubClass'].astype(int)
data_test['MSSubClass'] = data_test[['MSSubClass']].apply(cat_subclass, axis=1)
data_test['MSSubClass'] = data_test['MSSubClass'].astype(int)
#sns.factorplot('MSSubClass','SalePrice', data=data_train, size=4, aspect=3)
#sns.plt.show()



#Categorizing for Zoning
def cat_zone(zone):
    temp = str(zone)
    if temp == 'FV' or temp =='RL':
        return 2
    elif temp == 'C (all)':
        return 0
    else:
        return 1
data_train['MSZoning'] = data_train['MSZoning'].apply(cat_zone).astype(int)
data_test['MSZoning'] = data_test['MSZoning'].apply(cat_zone).astype(int)
#sns.factorplot('MSZoning','SalePrice', data=data_train, size=4, aspect=3)
#sns.plt.show()

def cat_contour(cont):
    temp = str(cont)
    if cont == 'Bnk':
        return 0
    elif cont == 'Low' or cont == 'HLS':
        return 2
    else:
        return 1
data_train['LandContour'] = data_train['LandContour'].apply(cat_contour).astype(int)
data_test['LandContour'] = data_test['LandContour'].apply(cat_contour).astype(int)
#sns.factorplot('LandContour','SalePrice', data=data_train, size=4, aspect=3)
#sns.plt.show()

def cat_util(util):
    if util == 'NoSeWa':
        return 0
    else:
        return 1
data_train['Utilities'] = data_train['Utilities'].apply(cat_util).astype(int)
data_test['Utilities'] = data_test['Utilities'].apply(cat_util).astype(int)

neighMap = {'NoRidge':5,
            'NridgHt':5,
            'StoneBr':5,
            'Veenker':4,
            'Crawfor':4,
            'Somerst':4,
            'Timber':4,
            'ClearCr':4,
            'CollgCr':3,
            'NWAmes':3,
            'SawyerW':3,
            'Gilbert':3,
            'Blmngtn':3,
            'Mitchel':2,
            'NAmes':2,
            'NPkVill':2,
            'SWISU':2,
            'Blueste':2,
            'OldTown':1,
            'BrkSide':1,
            'Sawyer':2,
            'Edwards':1,
            'IDOTRR':0,
            'MeadowV':0,
            'BrBale':0}
def cat_neigh(neigh):
    if neigh in neighMap.keys():
        return neighMap.get(neigh)
    else:
        return 3
data_train['Neighborhood'] = data_train['Neighborhood'].apply(cat_neigh).astype(int)
data_test['Neighborhood'] = data_test['Neighborhood'].apply(cat_neigh).astype(int)


sns.factorplot('BldgType','SalePrice', data=data_train, size=4, aspect=3)
sns.plt.show()
sns.factorplot('RoofMatl','SalePrice', data=data_train, size=4, aspect=3)
sns.plt.show()


print("Training data")
print("_____________")
for col in list(data_train):
    print (col, ":", data_train[col].isnull().sum())
print("Test data")
print("_____________")
for col in list(data_test):
    print (col, ":", data_test[col].isnull().sum())