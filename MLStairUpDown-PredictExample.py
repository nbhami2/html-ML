#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import pandas for dataframes, import csv, import os for file handling
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) #suppress futurewarnings from matplotlib
warnings.simplefilter(action='ignore', category=DeprecationWarning)

import pandas as pd
from pandas import read_csv
import math
from scipy import stats as st
import csv
import sklearn
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from xgboost import XGBRegressor

global df
csv_name = 'ExampleData.csv' #INPUT csv name here
df = pd.read_csv(csv_name) #import csv

for col in df.columns:
    if (('GlobalEfficiency' in col) or ('MaximizedModularity' in col) 
        or ('MeanClusteringCoeff' in col) or ('MeanTotalStrength'in col)
        or ('NetworkCharacteristic' in col) or ('TotalStrength' in col)
        or ('dummyrest' in col) or ('session_id' in col) or ('subject_id' in col)
        or ('dummy_rest' in col) or ('file_name' in col) or ('1back' in col)
        or ('acq_id' in col) or ('anatomical_zstat1' in col) or ('datetime' in col)):
        del df[col]

df['Sex'].replace(['Female','Male'],[0,1],inplace=True)

for col in df.columns:
    if(df[col].isnull().values.any()):
        if(df[col].isnull().sum()>50):
            del(df[col])
    
df=df.dropna()
df = df._get_numeric_data()


# In[ ]:


#Find best test/train split using Random Forest Regression models - not all features are known to be linear
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = df.drop('fft_stair_ds_tester1', axis=1)
y = df['fft_stair_ds_tester1']


# 25% test size
X_train_25, X_test_25, y_train_25, y_test_25 = train_test_split(X, y, test_size=0.25, random_state=42)
ss = StandardScaler()
X_train_25_scaled = ss.fit_transform(X_train_25)
X_test_25_scaled = ss.transform(X_test_25)

# 20% test size
X_train_20, X_test_20, y_train_20, y_test_20 = train_test_split(X, y, test_size=0.2, random_state=42)
ss = StandardScaler()
X_train_20_scaled = ss.fit_transform(X_train_20)
X_test_20_scaled = ss.transform(X_test_20)

# 15% test size
X_train_15, X_test_15, y_train_15, y_test_15 = train_test_split(X, y, test_size=0.15, random_state=42)
ss = StandardScaler()
X_train_15_scaled = ss.fit_transform(X_train_15)
X_test_15_scaled = ss.transform(X_test_15)

# 12.5% test size
X_train_125, X_test_125, y_train_125, y_test_125 = train_test_split(X, y, test_size=0.125, random_state=42)
ss = StandardScaler()
X_train_125_scaled = ss.fit_transform(X_train_125)
X_test_125_scaled = ss.transform(X_test_125)

#Linear Regression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

mae25, mse25 = [],[]
mae20, mse20 = [],[]
mae15, mse15 = [],[]
mae125, mse125 = [],[]
for i in range(0,10):
    model = RandomForestRegressor()
    model.fit(X_train_25,y_train_25)
    result = model.predict(X_test_25)
    mae = mean_absolute_error(y_test_25, result)
    mse = mean_squared_error(y_test_25, result)
    mae25.append(mae)
    mse25.append(mse)

    model = RandomForestRegressor()
    model.fit(X_train_20,y_train_20)
    result = model.predict(X_test_20)
    mae = mean_absolute_error(y_test_20, result)
    mse = mean_squared_error(y_test_20, result)
    mae20.append(mae)
    mse20.append(mse)

    model = RandomForestRegressor()
    model.fit(X_train_15,y_train_15)
    result = model.predict(X_test_15)
    mae = mean_absolute_error(y_test_15, result)
    mse = mean_squared_error(y_test_15, result)
    mae15.append(mae)
    mse15.append(mse)

    model = RandomForestRegressor()
    model.fit(X_train_125,y_train_125)
    result = model.predict(X_test_125)
    mae = mean_absolute_error(y_test_125, result)
    mse = mean_squared_error(y_test_125, result)
    mae125.append(mae)
    mse125.append(mse)
print('Average Mean Absolute Error (25%):'+str(sum(mae25)/len(mae25)))
print('Average Mean Squared Error (25%):'+str(sum(mse25)/len(mse25)))
print('Average Mean Absolute Error (20%):'+str(sum(mae20)/len(mae20)))
print('Average Mean Squared Error (20%):'+str(sum(mse20)/len(mse20)))
print('Average Mean Absolute Error (15%):'+str(sum(mae15)/len(mae15)))
print('Average Mean Squared Error (15%):'+str(sum(mse15)/len(mse15)))
print('Average Mean Absolute Error (12.5%):'+str(sum(mae125)/len(mae125)))
print('Average Mean Squared Error (12.5%):'+str(sum(mse125)/len(mse125)))

#15% test split yields lowest average errors across multiple runs


# In[ ]:


# X = df.iloc[:,0:df.columns.size].values
# Y = df['fft_stair_us_tester1'].values

#https://towardsdatascience.com/3-essential-ways-to-calculate-feature-importance-in-python-2f9149592155
#preprocess data, X and Y train/test split, scale x values 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = df.drop('fft_stair_ds_tester1', axis=1)
X2 = df.drop(['fft_stair_ds_tester1','Sex'], axis=1) #no sex
y = df['fft_stair_ds_tester1']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.15, random_state=42)

ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

ss2 = StandardScaler()
X2_train_scaled = ss2.fit_transform(X2_train)
X2_test_scaled = ss2.transform(X2_test)


# In[ ]:


#feature selection using XGB classifier ------ SEX included
model = XGBRegressor()
model.fit(X_train_scaled, y_train)
importances = pd.DataFrame(data={
    'Attribute': X_train.columns,
    'Importance': model.feature_importances_
})
importances = importances.sort_values(by='Importance', ascending=False)
importances.drop(importances.loc[importances['Importance']<0.007].index, inplace=True)
print(importances)
plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
plt.title('Feature importances, XGB, With Sex')
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:


#feature selection using XGB classifier ------ SEX NOT included

model = XGBRegressor()
model.fit(X2_train_scaled, y_train)
importances = pd.DataFrame(data={
    'Attribute': X2_train.columns,
    'Importance': model.feature_importances_
})
importances = importances.sort_values(by='Importance', ascending=False)
importances.drop(importances.loc[importances['Importance']<0.004].index, inplace=True)
print(importances)
plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
plt.title('Feature importances, XGB, Without Sex')
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:


#https://machinelearningmastery.com/feature-selection-for-regression-data/
#feature selection using SelectKBest, f_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

def select_features1(X_train, y_train, X_test):
    fs = SelectKBest(score_func=f_regression, k='all')
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs
 
X_train_fs, X_test_fs, fs = select_features1(X_train, y_train, X_test)
for i in range(len(fs.scores_)):
    if(fs.scores_[i]>10):
        print(df.columns[i]+": "+ str(fs.scores_[i]))
plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
plt.title('Feature importance, KBest (f_reg), with Sex')
plt.show()


# In[ ]:


#https://machinelearningmastery.com/feature-selection-for-regression-data/
#feature selection using SelectKBest, f_regression, sex Not Included
def select_features2(X_train, y_train, X_test):
    fs = SelectKBest(score_func=f_regression, k='all')
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs
 
X_train_fsNoSex, X_test_fsNoSex, fsNoSex = select_features2(X2_train, y_train, X2_test)
for i in range(len(fsNoSex.scores_)):
    if(fsNoSex.scores_[i]>10):
        print(df.columns[i]+": "+ str(fsNoSex.scores_[i]))
plt.bar([i for i in range(len(fsNoSex.scores_))], fsNoSex.scores_)
plt.title('Feature importance, KBest (f_reg), without Sex')
plt.show()


# In[ ]:


#https://machinelearningmastery.com/feature-selection-for-regression-data/
#feature selection using SelectKBest, mutual_info_regression
from sklearn.feature_selection import mutual_info_regression

def select_features3(X_train, y_train, X_test):
    fs = SelectKBest(score_func=mutual_info_regression, k='all')
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs
 
X_train_fs2, X_test_fs2, fs2 = select_features3(X_train, y_train, X_test)
for i in range(len(fs2.scores_)):
    if(fs2.scores_[i]>.175):
        print(df.columns[i]+": "+ str(fs2.scores_[i]))
plt.bar([i for i in range(len(fs2.scores_))], fs2.scores_)
plt.title('Feature importance, KBest (MutualInfo), with Sex')
plt.show()


# In[ ]:


#https://machinelearningmastery.com/feature-selection-for-regression-data/
#feature selection using SelectKBest, mutual_info_regression, Sex Not Included
from sklearn.feature_selection import mutual_info_regression

def select_features4(X_train, y_train, X_test):
    fs = SelectKBest(score_func=mutual_info_regression, k='all')
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs
 
X_train_fs2NoSex, X_test_fs2NoSex, fs2NoSex = select_features4(X2_train, y_train, X2_test)
for i in range(len(fs2NoSex.scores_)):
    if(fs2NoSex.scores_[i]>.175):
        print(df.columns[i]+": "+ str(fs2NoSex.scores_[i]))
plt.bar([i for i in range(len(fs2NoSex.scores_))], fs2NoSex.scores_)
plt.title('Feature importance, KBest (MutualInfo), without Sex')
plt.show()


# In[ ]:


#------------------------------------------------------------------------


# In[ ]:


#Linear Regression, All features (with Sex)
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

model = LinearRegression()
model.fit(X_train,y_train)
result = model.predict(X_test)
mae = mean_absolute_error(y_test, result)
mse = mean_squared_error(y_test, result)
print('Mean Absolute Error:'+str(mae))
print('Mean Squared Error:'+str(mse))


# In[ ]:


#Random Forest Regression, All features (with Sex)
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

model = RandomForestRegressor()
model.fit(X_train,y_train)
result = model.predict(X_test)
mae = mean_absolute_error(y_test, result)
mse = mean_squared_error(y_test, result)
print('Mean Absolute Error:'+str(mae))
print('Mean Squared Error:'+str(mse))


# In[ ]:


#linear regression with feature selections used above
def select(X_train, y_train, X_test):
    fs = SelectKBest(score_func=f_regression, k=10)
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs
 
model = LinearRegression()
model.fit(X_train_fs, y_train)
result1 = model.predict(X_test_fs)
mae1 = mean_absolute_error(y_test, result1)
mse1 = mean_squared_error(y_test, result1)
print('MAE1: %.3f' % mae1)
print('MSE1: %.3f' % mse1)

model = LinearRegression()
model.fit(X_train_fsNoSex, y_train)
result2 = model.predict(X_test_fsNoSex)
mae2 = mean_absolute_error(y_test, result2)
mse2 = mean_squared_error(y_test, result2)
print('MAE2: %.3f' % mae2)
print('MSE2: %.3f' % mse2)

model = LinearRegression()
model.fit(X_train_fs2, y_train)
result3 = model.predict(X_test_fs2)
mae3 = mean_absolute_error(y_test, result3)
mse3 = mean_squared_error(y_test, result3)
print('MAE3: %.3f' % mae3)
print('MSE3: %.3f' % mse3)

model = LinearRegression()
model.fit(X_train_fs2NoSex, y_train)
result4 = model.predict(X_test_fs2NoSex)
mae4 = mean_absolute_error(y_test, result4)
mse4 = mean_squared_error(y_test, result4)
print('MAE4: %.3f' % mae4)
print('MSE4: %.3f' % mse4)


# In[ ]:


#Random Forest regression with feature selections used above
def select(X_train, y_train, X_test):
    fs = SelectKBest(score_func=f_regression, k=10)
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs
 
model = RandomForestRegressor()
model.fit(X_train_fs, y_train)
result1 = model.predict(X_test_fs)
mae1 = mean_absolute_error(y_test, result1)
mse1 = mean_squared_error(y_test, result1)
print('MAE1: %.3f' % mae1)
print('MSE1: %.3f' % mse1)

model = RandomForestRegressor()
model.fit(X_train_fsNoSex, y_train)
result2 = model.predict(X_test_fsNoSex)
mae2 = mean_absolute_error(y_test, result2)
mse2 = mean_squared_error(y_test, result2)
print('MAE2: %.3f' % mae2)
print('MSE2: %.3f' % mse2)

model = RandomForestRegressor()
model.fit(X_train_fs2, y_train)
result3 = model.predict(X_test_fs2)
mae3 = mean_absolute_error(y_test, result3)
mse3 = mean_squared_error(y_test, result3)
print('MAE3: %.3f' % mae3)
print('MSE3: %.3f' % mse3)

model = RandomForestRegressor()
model.fit(X_train_fs2NoSex, y_train)
result4 = model.predict(X_test_fs2NoSex)
mae4 = mean_absolute_error(y_test, result4)
mse4 = mean_squared_error(y_test, result4)
print('MAE4: %.3f' % mae4)
print('MSE4: %.3f' % mse4)


# In[ ]:


structural = ['fwhm','snr','cnr','fber','efc','qi1','qi2','icvs','rpve','inu','summary']
functional = ['efc','fber','fwhm','ghost_x','snr','dvars','gcor','mean_fd','num_fd','perc_fd','outlier','quality']
X_s = X_f = X.copy(deep=True)
for col in X_s.columns:
    flag = True
    for item in structural:
        if (item in col):
            flag = False
    if(flag):
        del X_s[col]
        
X_f = df.copy(deep=True)
for col in X_f.columns:
    flag = True
    for item in functional:
        if (item in col):
            flag = False
    if(flag):
        del X_f[col]

display(X_s)
y = df['fft_stair_ds_tester1']
X_s_train, X_s_test, y_s_train, y_s_test = train_test_split(X_s, y, test_size=0.2, random_state=42)
X_f_train, X_f_test, y_f_train, y_f_test = train_test_split(X_f, y, test_size=0.2, random_state=42)
display(X_s_train)

ss = StandardScaler()
X_s_train_scaled = ss.fit_transform(X_s_train)
X_s_test_scaled = ss.transform(X_s_test)

ss2 = StandardScaler()
X_f_train_scaled = ss2.fit_transform(X_f_train)
X_f_test_scaled = ss2.transform(X_f_test)


# In[ ]:


#feature selection using XGB classifier ------ structural and functional
model = XGBRegressor()
model.fit(X_s_train_scaled, y_s_train)
importances_s = pd.DataFrame(data={
    'Attribute': X_s_train.columns,
    'Importance': model.feature_importances_
})
importances_s = importances_s.sort_values(by='Importance', ascending=False)
importances_s.drop(importances_s.loc[importances_s['Importance']<0.01].index, inplace=True)
print(importances_s.head())
plt.bar(x=importances_s['Attribute'], height=importances_s['Importance'], color='#087E8B')
plt.title('Feature importances, XGB, Structural')
plt.xticks(rotation='vertical')
plt.show()

model = XGBRegressor()
model.fit(X_f_train_scaled, y_f_train)
importances_f = pd.DataFrame(data={
    'Attribute': X_f_train.columns,
    'Importance': model.feature_importances_
})
importances_f = importances_f.sort_values(by='Importance', ascending=False)
importances_f.drop(importances_f.loc[importances_f['Importance']<0.01].index, inplace=True)
print(importances_f.head())
plt.bar(x=importances_f['Attribute'], height=importances_f['Importance'], color='#087E8B')
plt.title('Feature importances, XGB, functional')
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:


#https://machinelearningmastery.com/feature-selection-for-regression-data/
#Regression with tuned number of features - structural
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define the pipeline to evaluate
model = LinearRegression()
fs = SelectKBest(score_func=mutual_info_regression)
pipeline = Pipeline(steps=[('sel',fs), ('lr', model)])
# define the grid
grid = dict()
grid['sel__k'] = [i for i in range(X_s_train.shape[1]-(X_s_train.shape[1]-1), X_s_train.shape[1]+1)]
# define the grid search
search = GridSearchCV(pipeline, grid, scoring='neg_mean_squared_error', n_jobs=-1, cv=cv)
# perform the search
results = search.fit(X_s_train, y_s_train)
# summarize best
print('Best MAE: %.3f' % results.best_score_)
print('Best Config: %s' % results.best_params_)
print("Actual    Predicted")
count = 0
for i in y_s_test:
    print(str(i)+"  "+str(results.predict(X_s_test)[count]))
    count+=1
# summarize all
means = results.cv_results_['mean_test_score']
params = results.cv_results_['params']
print("\nAll configs:")
for mean, param in zip(means, params):
    print(">%.3f with: %r" % (mean, param))


# In[ ]:


#https://machinelearningmastery.com/feature-selection-for-regression-data/
#Regression with tuned number of features - functional
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define the pipeline to evaluate
model = LinearRegression()
fs = SelectKBest(score_func=mutual_info_regression)
pipeline = Pipeline(steps=[('sel',fs), ('lr', model)])
# define the grid
grid = dict()
grid['sel__k'] = [i for i in range(X_f_train.shape[1]-(X_f_train.shape[1]-1), X_f_train.shape[1]+1)]
# define the grid search
search = GridSearchCV(pipeline, grid, scoring='neg_mean_squared_error', n_jobs=-1, cv=cv)
# perform the search
results = search.fit(X_f_train, y_f_train)
# summarize best
print('Best MAE: %.3f' % results.best_score_)
print('Best Config: %s' % results.best_params_)
print("Actual    Predicted")
count = 0
for i in y_f_test:
    print(str(i)+"  "+str(results.predict(X_f_test)[count]))
    count+=1
# summarize all
means = results.cv_results_['mean_test_score']
params = results.cv_results_['params']
print("\nAll configs:")
for mean, param in zip(means, params):
    print(">%.3f with: %r" % (mean, param))


# In[ ]:


#https://machinelearningmastery.com/feature-selection-for-regression-data/
#Regression with tuned number of features - functional
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define the pipeline to evaluate
model = LinearRegression()
fs = SelectKBest(score_func=mutual_info_regression)
pipeline = Pipeline(steps=[('sel',fs), ('lr', model)])
# define the grid
grid = dict()
grid['sel__k'] = [i for i in range(X_train.shape[1]-(X_train.shape[1]-1), X_train.shape[1]+1)]
# define the grid search
search = GridSearchCV(pipeline, grid, scoring='neg_mean_squared_error', n_jobs=-1, cv=cv)
# perform the search
results = search.fit(X_train, y_train)
# summarize best
print('Best MAE: %.3f' % results.best_score_)
print('Best Config: %s' % results.best_params_)
print("Actual    Predicted")
count = 0
for i in y_test:
    print(str(i)+"  "+str(results.predict(X_test)[count]))
    count+=1
# summarize all
means = results.cv_results_['mean_test_score']
params = results.cv_results_['params']
print("\nAll configs:")
for mean, param in zip(means, params):
    print(">%.3f with: %r" % (mean, param))


# In[ ]:


#https://machinelearningmastery.com/feature-selection-for-regression-data/
#Regression with tuned number of features NO SEX
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define the pipeline to evaluate
model = LinearRegression()
fs = SelectKBest(score_func=mutual_info_regression)
pipeline = Pipeline(steps=[('sel',fs), ('lr', model)])
# define the grid
grid = dict()
grid['sel__k'] = [i for i in range(X2_train.shape[1]-(X2_train.shape[1]-1), X2_train.shape[1]+1)]
# define the grid search
search = GridSearchCV(pipeline, grid, scoring='neg_mean_squared_error', n_jobs=-1, cv=cv)
# perform the search
results = search.fit(X2_train, y2_train)
# summarize best
print('Best MAE: %.3f' % results.best_score_)
print('Best Config: %s' % results.best_params_)
# summarize all
means = results.cv_results_['mean_test_score']
params = results.cv_results_['params']
for mean, param in zip(means, params):
    print(">%.3f with: %r" % (mean, param))


# In[ ]:


print("Actual    Predicted")
count = 0
for i in y2_test:
    print(str(i)+"  "+str(results.predict(X2_test)[count]))
    count+=1


# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeRegressor
rfe = RFE(
    estimator=DecisionTreeRegressor(),
    n_features_to_select=3,
)
model = DecisionTreeRegressor()
pipeline = Pipeline(steps=[('s',rfe),('m',model)])
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(pipeline, X_s_train, y_s_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
pipeline.fit(X_s_train, y_s_train)
pipeline.predict(X_s_test)
# count = 0
# for i in y_test:
#     print(str(i)+"  "+str(pipeline.predict(X_s_test)[count]))
#     count+=1
for i in range(X.shape[1]):
	print('Column: %d, Selected %s, Rank: %.3f' % (i, rfe.support_[i], rfe.ranking_[i]))


# In[ ]:


#https://machinelearningmastery.com/rfe-feature-selection-in-python/
#code pulled from above website to get best model from Logistic, Perceptron, Decision Tree, Random Forest, and Grad. Boost
#evaluated using cross validation between different mdoels

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
def get_models():
	models = dict()
	# lr
	rfe = RFE(estimator=LogisticRegression(), n_features_to_select=5)
	model = DecisionTreeRegressor()
	models['lr'] = Pipeline(steps=[('s',rfe),('m',model)])
	# perceptron
	rfe = RFE(estimator=Perceptron(), n_features_to_select=5)
	model = DecisionTreeRegressor()
	models['per'] = Pipeline(steps=[('s',rfe),('m',model)])
	# cart
	rfe = RFE(estimator=DecisionTreeRegressor(), n_features_to_select=5)
	model = DecisionTreeRegressor()
	models['cart'] = Pipeline(steps=[('s',rfe),('m',model)])
	# rf
	rfe = RFE(estimator=RandomForestRegressor(), n_features_to_select=5)
	model = DecisionTreeRegressor()
	models['rf'] = Pipeline(steps=[('s',rfe),('m',model)])
	# gbm
	rfe = RFE(estimator=GradientBoostingRegressor(), n_features_to_select=5)
	model = DecisionTreeRegressor()
	models['gbm'] = Pipeline(steps=[('s',rfe),('m',model)])
	return models
 
def evaluate_model(model, X, y):
	cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(pipeline, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
	return scores

models = get_models()

results, names = list(), list()
for name, model in models.items():
    scores = evaluate_model(model, X, y)
    results.append(scores)
    names.append(name)
    print(scores)
    print(name+" "+mean(scores)+" "+ std(scores))

