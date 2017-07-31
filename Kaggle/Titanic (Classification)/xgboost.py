import csv
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import warnings
# xgboost <= 0.6a2 shows a warning when used with scikit-learn 0.18+
warnings.filterwarnings('ignore', category=DeprecationWarning)
import xgboost as xgb
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")

def changeNames(data):
    for i in range(len(data['Name'])):
        name = data['Name'][i]
        ind1 = name.index(',')
        ind2 = name.index('.')
        title = name[ind1+2:ind2]
        if title == 'Mme' or title == 'Ms' or title == 'Lady' or title == 'Sir' or title == 'Mlle' or title == 'the Countess':
            data['Name'][i] = 2
        elif title == 'Mrs' or title == 'Miss'or title == 'Master' or title == 'Dr' or title == 'Major' or title == 'Col':
            data['Name'][i] = 1
        else:
            data['Name'][i] = 0

changeNames(data_train)
changeNames(data_test)
data_train['Name'] = data_train['Name'].astype(int)
data_test['Name'] = data_test['Name'].astype(int)

# Fill in missing age for both training and test data
average_age_titanic   = data_train["Age"].mean()
std_age_titanic       = data_train["Age"].std()
count_nan_age_titanic = data_train["Age"].isnull().sum()
average_age_test   = data_test["Age"].mean()
std_age_test       = data_test["Age"].std()
count_nan_age_test = data_test["Age"].isnull().sum()
rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)
rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)
data_train['Age'].dropna().astype(int)
data_test['Age'].dropna().astype(int)
data_train["Age"][np.isnan(data_train["Age"])] = rand_1
data_test["Age"][np.isnan(data_test["Age"])] = rand_2
data_train['Age'] = data_train['Age'].astype(int)
data_test['Age'] = data_test['Age'].astype(int)

#Fill in missing fare for test data
data_test["Fare"].fillna(data_test["Fare"].median(), inplace=True)

#Fill in missing embarked
data_train["Embarked"] = data_train["Embarked"].fillna("S")
#Maps embarked values to integers in order of survived probability
data_train['Embarked'] = data_train['Embarked'].map({'S': 0, 'Q': 1,'C': 2}).astype(int)
data_test['Embarked'] = data_test['Embarked'].map({'S': 0, 'Q': 1,'C': 2}).astype(int)

#Convert to family
data_train['Family'] =  data_train["Parch"] + data_train["SibSp"]
data_train['Family'].loc[data_train['Family'] > 0] = 1
data_train['Family'].loc[data_train['Family'] == 0] = 0

data_test['Family'] =  data_test["Parch"] + data_test["SibSp"]
data_test['Family'].loc[data_test['Family'] > 0] = 1
data_test['Family'].loc[data_test['Family'] == 0] = 0

#Convert to sex
def get_person(passenger):
    age, sex = passenger
    return 'child' if age < 16 else sex


data_train['Person'] = data_train[['Age', 'Sex']].apply(get_person, axis=1)
data_test['Person'] = data_test[['Age', 'Sex']].apply(get_person, axis=1)
data_train['Person'] = data_train['Person'].map({'male': 0, 'child': 1,'female': 2}).astype(int)
data_test['Person'] = data_test['Person'].map({'male': 0, 'child': 1,'female': 2}).astype(int)

data_train = data_train.drop(['PassengerId','Ticket','Cabin','SibSp','Parch','Age','Sex'], axis=1)
data_test = data_test.drop(['PassengerId','Ticket','Cabin','SibSp','Parch','Age','Sex'], axis=1)

X_train = data_train.drop(['Survived'], axis=1)
Y_train = data_train['Survived']
X_test  = data_test.copy()

class CSCTransformer:
    def transform(self, xs):
        # work around https://github.com/dmlc/xgboost/issues/1238#issuecomment-243872543
        return xs.tocsc()
    def fit(self, *args):
        return self

clf = xgb.XGBClassifier()
vec = DictVectorizer()
pipeline = make_pipeline(vec, CSCTransformer(), clf)

def evaluate(_clf):
    _clf.fit(X_train, Y_train)  # so that parts of the original pipeline are fitted

evaluate(pipeline)
y = pipeline.predict(X_test)
print(y)