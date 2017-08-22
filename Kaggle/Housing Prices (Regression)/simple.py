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
data_train = data_train.drop(['Id','Street','MiscFeature','Alley','Fence'],axis=1)
data_test = data_test.drop(['Id','Street','MiscFeature','Alley','Fence'],axis=1)

data_train = data_train.drop(data_train[(data_train['GrLivArea']>4000) & (data_train['SalePrice']<300000)].index)


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



def cat_type(bldg):
    if bldg == '1Fam' or bldg == 'TwnhsE':
        return 2
    elif bldg != '':
        return 0
    else:
        return 1
data_train['BldgType'] = data_train['BldgType'].apply(cat_type).astype(int)
data_test['BldgType'] = data_test['BldgType'].apply(cat_type).astype(int)

def cat_masvnr(mas):
    if mas == 'Stone':
        return 2
    elif mas == 'BrkFace':
        return 1
    else:
        return 0
data_train['MasVnrType'] = data_train['MasVnrType'].apply(cat_masvnr).astype(int)
data_test['MasVnrType'] = data_test['MasVnrType'].apply(cat_masvnr).astype(int)

map_qualna = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0}
map_qual = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':2}
data_train['ExterQual'] = data_train['ExterQual'].map(map_qual).astype(int)
data_test['ExterQual'] = data_test['ExterQual'].map(map_qual).astype(int)

data_train['BsmtCond'] = data_train['BsmtCond'].fillna('NA')
data_test['BsmtCond'] = data_test['BsmtCond'].fillna('NA')
data_train['BsmtQual'] = data_train['BsmtQual'].fillna('NA')
data_test['BsmtQual'] = data_test['BsmtQual'].fillna('NA')

data_train['BsmtCond'] = data_train['BsmtCond'].map(map_qual).astype(int)
data_test['BsmtCond'] = data_test['BsmtCond'].map(map_qual).astype(int)
data_train['BsmtQual'] = data_train['BsmtQual'].map(map_qualna).astype(int)
data_test['BsmtQual'] = data_test['BsmtQual'].map(map_qualna).astype(int)


data_train['BsmtFinType1'] = data_train['BsmtFinType1'].fillna('NA')
data_test['BsmtFinType1'] = data_test['BsmtFinType1'].fillna('NA')
def cat_bsmtfin1(type):
    if type == 'GLQ':
        return 3
    elif type == 'ALQ' or type == 'Unf':
        return 2
    elif type == 'NA':
        return 0
    else:
        return 1
data_train['BsmtFinType1'] = data_train['BsmtFinType1'].apply(cat_bsmtfin1).astype(int)
data_test['BsmtFinType1'] = data_test['BsmtFinType1'].apply(cat_bsmtfin1).astype(int)


def cat_poolqc(qual):
    if qual == 'Ex':
        return 2;
    elif qual == 'None':
        return 0;
    else:
        return 1;
data_train['PoolQC'] = data_train['PoolQC'].fillna("None")
data_test['PoolQC'] = data_test['PoolQC'].fillna("None")
data_train['PoolQC'] = data_train['PoolQC'].apply(cat_poolqc).astype(int)
data_test['PoolQC'] = data_test['PoolQC'].apply(cat_poolqc).astype(int)


def cat_paved(pave):
    if pave == 'Y':
        return 2;
    elif pave == 'N':
        return 0;
    else:
        return 1;
data_train['PavedDrive'] = data_train['PavedDrive'].apply(cat_paved).astype(int)
data_test['PavedDrive'] = data_test['PavedDrive'].apply(cat_paved).astype(int)

map_qualFP = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':1}
data_train['FireplaceQu'] = data_train['FireplaceQu'].fillna("NA")
data_test['FireplaceQu'] = data_test['FireplaceQu'].fillna("NA")
data_train['FireplaceQu'] = data_train['FireplaceQu'].map(map_qualFP).astype(int)
data_test['FireplaceQu'] = data_test['FireplaceQu'].map(map_qualFP).astype(int)


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


