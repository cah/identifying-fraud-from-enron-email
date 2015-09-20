import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

## Determine Best features
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif,f_regression

## Classifiers 
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB

## Pipeline
from sklearn.pipeline import Pipeline, FeatureUnion 

## Scalers
from sklearn.preprocessing import StandardScaler

## Component Reduction
from sklearn.lda import LDA
from check import output_classifier

## Scoring 
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics.scorer import make_scorer

## Split and validation
from sklearn.cross_validation import StratifiedShuffleSplit

## Search for best parameters for classifier
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

def score_func(y_true, y_pred, **kwargs):
    r = recall_score(y_true, y_pred, **kwargs)
    p = precision_score(y_true, y_pred, **kwargs)
    if r > 0.30 and p > 0.30:
       return f1_score(y_true, y_pred, **kwargs)
    else:
       return 0

scorer       = make_scorer(score_func)


data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

df = pd.DataFrame.from_dict(data_dict)
df = df.transpose()
df.describe()

df = df.replace('NaN', np.nan) 
df=df.dropna(thresh=4,axis=0)

df.drop('email_address', 1,inplace=True)
df = df.convert_objects(convert_numeric=True)
df = df.apply(lambda x: x.fillna(x.mean()),axis=0)


df.drop('TOTAL', 0,inplace=True)
df['to_poi_ratio']   = df.from_this_person_to_poi/df.from_messages


features_list = ['poi']
all_features = list(df)
all_features.remove('poi')
features_list.extend(all_features)

my_dataset = df.T.to_dict('dict')
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

clf = Pipeline(steps=[('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)),
 ('kbest', SelectKBest(k=5, score_func=f_classif)),
 ('lda', LDA(n_components=1, priors=None, shrinkage=None, solver='svd',
  store_covariance=False, tol=0.0001)), ('classifier', GaussianNB())])

test_classifier(clf, my_dataset, features_list,folds=1000)

dump_classifier_and_data(clf, my_dataset, features_list)
