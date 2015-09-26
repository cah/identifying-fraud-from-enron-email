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
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import MiniBatchKMeans, KMeans

## Pipeline
from sklearn.pipeline import Pipeline, FeatureUnion 

## Scalers
from sklearn.preprocessing import StandardScaler,MinMaxScaler

## Component Reduction
from sklearn.lda import LDA
from check import output_classifier
from ModelTransformer import ModelTransformer

## Scoring 
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics.scorer import make_scorer

## Split and validation
from sklearn.cross_validation import StratifiedShuffleSplit

## Search for best parameters for classifier
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV


## Getting the data
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
df = pd.DataFrame.from_dict(data_dict)
df = df.transpose()


## Cleaning the data
df = df.replace('NaN', np.nan) 
df=df.dropna(thresh=4,axis=0)
df.drop('email_address', 1,inplace=True)
df = df.convert_objects(convert_numeric=True)
df = df.apply(lambda x: x.fillna(x.mean()),axis=0)
df.drop('TOTAL', 0,inplace=True)

## Creating new variable
df['to_poi_ratio']   = df.from_this_person_to_poi/df.from_messages


## Pull all columns names in feature list
features_list = ['poi']
all_features = list(df)
all_features.remove('poi')
features_list.extend(all_features)

## Create features and labels
my_dataset = df.T.to_dict('dict')
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

## Define scoring method to return 
## f1 when recall and precision are > 30 
def score_func(y_true, y_pred, **kwargs):
    r = recall_score(y_true, y_pred, **kwargs)
    p = precision_score(y_true, y_pred, **kwargs)
    if r > 0.30 and p > 0.30:
       return f1_score(y_true, y_pred, **kwargs)
    else:
       return 0

scorer       = make_scorer(score_func)

clf = Pipeline(steps=[
   # ('scaler', MinMaxScaler()),
   # ('features', FeatureUnion([
   #    ('ngram_tf_idf', Pipeline([
   #      ('kbest', SelectKBest(k=5, score_func=f_classif)),
   #      ('lda', LDA(n_components=1, priors=None, shrinkage=None, solver='svd', store_covariance=False, tol=0.0001)), 
   #      ('kmeans', MiniBatchKMeans(n_clusters=20, n_init=10, max_no_improvement=10, verbose=0)),
   #    ]))])),
   ('kbest', SelectKBest(k=5, score_func=f_classif)),
   ('lda', LDA(n_components=1, priors=None, shrinkage=None, solver='svd', store_covariance=False, tol=0.0001)), 
   ('kmeans', MiniBatchKMeans(n_clusters=20, n_init=10, max_no_improvement=10, verbose=0)),
   ('classifier', GaussianNB()) 
   ])

test_classifier(clf, my_dataset, features_list,folds=1000)

dump_classifier_and_data(clf, my_dataset, features_list)
