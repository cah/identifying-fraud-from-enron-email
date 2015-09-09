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
from sklearn.decomposition import PCA
from check import output_classifier



## Search for best parameters for classifier
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
pd.set_option('display.notebook_repr_html', True)


# Read the data and convert the data into pandas data frame. 

# In[108]:


data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
df = pd.DataFrame.from_dict(data_dict)
df = df.transpose()


# Drop all employees where all features expect poi are NaN.

# In[109]:

df = df.replace('NaN', np.nan) 
df=df.dropna(thresh=2,axis=0)


# Describe provides concise overview of the dataset. The first item that jumps out... The top value in each column is 'NaN' which means the dataset contains a significant amount of missing data. 

# In[110]:

df.email_address.head(5)


# Email_address column is unquie like the employees name and doesn't provide additional value. Therefore, I choose to drop the email_address field. 

# In[111]:

df.drop('email_address', 1,inplace=True)


# Let's convert the remaining fields to numbers and fill any NaNs with the mean. 

# In[112]:

df = df.convert_objects(convert_numeric=True)
df = df.apply(lambda x: x.fillna(x.mean()),axis=0)


# In[113]:

column='salary'
print df[[column]].sort([column], ascending=[0]).head(3)
plt.figure(figsize=(20,8))
df[column].plot(style='.')
x = range(len(df[column]))
plt.xticks(x,df.index)
locs, labels = plt.xticks()
print labels
plt.setp(labels, rotation=90)
plt.title(column)
plt.show()


# After plotting the employee salaries, it's clear the Total is an aggregate and it should be dropped as well. 

# In[114]:

df.drop('TOTAL', 0,inplace=True)


# In[115]:

column='salary'
print df[[column]].sort([column], ascending=[0]).head(3)
print df[[column]].sort([column], ascending=[0]).tail(3)
plt.figure(figsize=(20,8))
df[column].plot(style='.')
x = range(len(df[column]))
plt.xticks(x,df.index)
locs, labels = plt.xticks()
print labels
plt.setp(labels, rotation=90)
plt.title(column)
plt.show()



# In the salary plot, I notice 'THE TRAVEL AGENCY IN THE PARK' is not an employee. 

# In[116]:

df.drop('THE TRAVEL AGENCY IN THE PARK', 0,inplace=True)


## Uncomment to plot all features.... This was used to review and eliminate outliers. 

# for column in list(df):
#     if column not in ['email_address','poi','director_fees']:
#         if column not in ['email_address','poi']:
#             print df[[column]].sort([column], ascending=[0]).head(3)
#             plt.figure(figsize=(20,8))
#             df[column].plot(style='.')
#             x = range(len(df[column]))
#             plt.xticks(x,df.index)
#             locs, labels = plt.xticks()
#             print labels
#             plt.setp(labels, rotation=90)
#             plt.title(column)
#             plt.show()

#             plt.figure(figsize=(24,8))
#             plt.hist(df[column])
#             plt.title(column)
#             plt.show()




df['to_poi_ratio'] = df['from_this_person_to_poi']/df['from_messages']
my_dataset = df.T.to_dict('dict')

features_list = [
 'bonus',
 'deferral_payments',
 'deferred_income',
 'director_fees',
 'exercised_stock_options',
 'expenses',
 'from_messages',
 'from_poi_to_this_person',
 'from_this_person_to_poi',
 'loan_advances',
 'long_term_incentive',
 'other',
 'poi',
 'restricted_stock',
 'restricted_stock_deferred',
 'salary',
 'shared_receipt_with_poi',
 'to_messages',
 'total_payments',
 'total_stock_value',
 'to_poi_ratio'
]



### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

selector = SelectKBest(k='all').fit(features, labels).scores_
for score, feature in sorted(zip(selector, features_list[1:]), reverse=True):
    print feature, score


# Removed feature with impact less than 1 
# Removed: from_messages, director_fees,restricted_stock_deferred

# In[119]:

features_list = [
             'poi',
             'bonus',
             'deferral_payments',
             'deferred_income',
             #'director_fees',
             'exercised_stock_options',
             'expenses',
             #'from_messages',
             'from_poi_to_this_person',
             'from_this_person_to_poi',
             'loan_advances',
             'long_term_incentive',
             'other',
             'restricted_stock',
             #'restricted_stock_deferred',
             'salary',
             'shared_receipt_with_poi',
             'to_messages',
             'total_payments',
             'total_stock_value'
                ]



### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


dict_all = {}
dict_pca = {}
dict_lda = {}
names = [
   "Linear_SVC",
   "Naive_Bayes"]

classifiers = [
    LinearSVC(),
    GaussianNB()]


# In[121]:

print "With StandardScaler"
for name, clf in zip(names, classifiers):
    clf_all = Pipeline(steps=[
       ('scaler', StandardScaler()),
       ('classification', clf)
    ])
    dict_all[name] =  output_classifier(clf_all, my_dataset, features_list)
all = pd.DataFrame.from_dict(dict_all, orient='index')
all 


# In[122]:

print "With StandardScaler and PCA"
for name, clf in zip(names, classifiers):
    clf_pca = Pipeline(steps=[
       ('scaler', StandardScaler()),
       ('reduce_dim', PCA(n_components=1)),
       ('classification', clf)
     ])
    dict_pca[name] =  output_classifier(clf_pca, my_dataset, features_list)
pca = pd.DataFrame.from_dict(dict_pca, orient='index')
pca


# In[123]:

print "With StandardScaler and LDA"
for name, clf in zip(names, classifiers):
    clf_lda = Pipeline(steps=[
       ('scaler', StandardScaler()),
       ('reduce_dim', LDA(n_components=1)),
       ('classification', clf)
    ])
    dict_lda[name] =  output_classifier(clf_lda, my_dataset, features_list)
lda = pd.DataFrame.from_dict(dict_lda, orient='index')
lda


clf = Pipeline([
    ('scaler', StandardScaler()),
    ('kbest', SelectKBest()),
    ('lda', LDA()),
    ('classification',GaussianNB() ),
])


parameters = {
    'kbest__score_func'         : (f_classif, f_regression),
    'kbest__k'                  : range(5,len(features_list)),
    'lda__n_components'         : (1,2,3,4),
    'lda__store_covariance'     : (True, False), 
    'lda__tol'                  : (0.0001,0.001)
}

grid = GridSearchCV(clf,parameters,n_jobs=-1)
grid.fit(features, labels)
clf = grid.best_estimator_

print clf


selector = SelectKBest(k=5,score_func=f_classif).fit(features, labels).scores_
for score, feature in sorted(zip(selector, features_list[1:]), reverse=True):
    print feature


features_list = [
             'poi',
             'exercised_stock_options',
             'total_stock_value',
             'to_poi_ratio',
             'bonus',
             'salary'
     ]


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# In[131]:

### Dump your classifier, dataset, and features_list so
### anyone can run/check your results.

clf = Pipeline(steps=[('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)),
                      ('lda', LDA(n_components=1, priors=None, shrinkage=None, solver='svd',
  store_covariance=True, tol=0.0001)), ('classification', GaussianNB())])

test_classifier(clf, my_dataset, features_list,folds=1000)

dump_classifier_and_data(clf, my_dataset, features_list)

