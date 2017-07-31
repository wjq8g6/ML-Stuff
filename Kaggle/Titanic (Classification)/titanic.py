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

#isolates the title from the rest of the names
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

data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")

#Maps titles to integers in order of survived probability
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
'''
data_train['Family'] =  data_train["Parch"] + data_train["SibSp"]
data_train['Family'].loc[data_train['Family'] > 0] = 1
data_train['Family'].loc[data_train['Family'] == 0] = 0

data_test['Family'] =  data_test["Parch"] + data_test["SibSp"]
data_test['Family'].loc[data_test['Family'] > 0] = 1
data_test['Family'].loc[data_test['Family'] == 0] = 0
'''
#Convert to sex
def get_person(passenger):
    age, sex = passenger
    return 'child' if age < 17 else sex


data_train['Person'] = data_train[['Age', 'Sex']].apply(get_person, axis=1)
data_test['Person'] = data_test[['Age', 'Sex']].apply(get_person, axis=1)
data_train['Person'] = data_train['Person'].map({'male': 0, 'child': 1,'female': 2}).astype(int)
data_test['Person'] = data_test['Person'].map({'male': 0, 'child': 1,'female': 2}).astype(int)

data_train = data_train.drop(['PassengerId','Ticket','Cabin','Age','Sex'], axis=1)
data_test = data_test.drop(['PassengerId','Ticket','Cabin','Age','Sex'], axis=1)


print("Training data")
print("_____________")
for col in list(data_train):
    print (col, ":", data_train[col].isnull().sum())
print("Test data")
print("_____________")
for col in list(data_test):
    print (col, ":", data_test[col].isnull().sum())


X_train = data_train.drop(['Survived'], axis=1)
Y_train = data_train['Survived']
X_test  = data_test.copy()

'''
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
'''
#'''
grad_boost = GradientBoostingClassifier(n_estimators=100,learning_rate=0.1)
grad_boost.fit(X_train, Y_train)
Y_pred = grad_boost.predict(X_test)
grad_boost.score(X_train, Y_train)
#'''
'''
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
knn.score(X_train, Y_train)
'''
'''
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
svc.score(X_train, Y_train)
'''

newArr = []
newArr.append(['PassengerId','Survived'])
with open('pred.csv','w',newline='') as file:
    cwr = csv.writer(file, delimiter=',')
    count = 892
    for k in Y_pred:
        blank = []
        blank.append(count)
        blank.append(int(k))
        newArr.append(blank)
        count+=1
    #np.savetxt(file, newArr, delimiter=',')
    cwr.writerows(newArr)