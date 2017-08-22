import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from scipy.stats import skew
from scipy.stats.stats import pearsonr

all_data = pd.read_csv('cerv_cancer.csv')
