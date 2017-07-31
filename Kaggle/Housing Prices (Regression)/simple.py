import csv
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from scipy.stats import skew
from scipy.stats.stats import pearsonr

data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
data_train = data_train.drop(['Id','Street','PoolQC','MiscFeature','Alley','Fence','FireplaceQu'],axis=1)
data_test = data_test.drop(['Id','Street','PoolQC','MiscFeature','Alley','Fence','FireplaceQu'],axis=1)

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


#Categorizing for Zoning

def cat_zone(zone):
    temp = str(zone)
    if temp == 'FV' or temp =='RL':
        return 2
    elif temp == 'C (all)':
        return 0
    else:
        return 1
data_train['MSZoning'] = data_train['MSZoning'].fillna('RM')
data_test['MSZoning'] = data_test['MSZoning'].fillna('RM')
data_train['MSZoning'] = data_train['MSZoning'].apply(cat_zone).astype(int)
data_test['MSZoning'] = data_test['MSZoning'].apply(cat_zone).astype(int)


#Categorizing for Contour
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

#Neighborhood Cat might not help
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

print("Training data")
print("_____________")
for col in list(data_train):
    print (col, ":", data_train[col].isnull().sum())
print("Test data")
print("_____________")
for col in list(data_test):
    print (col, ":", data_test[col].isnull().sum())

all_data = pd.concat((data_train.loc[:,'MSSubClass':'SaleCondition'],
                      data_test.loc[:,'MSSubClass':'SaleCondition']))


data_train['SalePrice'] = np.log1p(data_train['SalePrice'])

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = data_train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())

Y_train = data_train['SalePrice']
X_train = all_data[:data_train.shape[0]]
X_test = all_data[data_train.shape[0]:]


def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, X_train, Y_train, scoring="neg_mean_squared_error", cv=5))
    return (rmse)




model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, Y_train)
print(rmse_cv(model_lasso).mean())


pred = np.exp(model_lasso.predict(X_test)) - 1
newArr = []
newArr.append(['Id','SalePrice'])
with open('pred.csv','w',newline='') as file:
    cwr = csv.writer(file, delimiter=',')
    count = 1461
    for k in pred:
        blank = []
        blank.append(count)
        blank.append(k)
        newArr.append(blank)
        count+=1
    #np.savetxt(file, newArr, delimiter=',')
    cwr.writerows(newArr)


