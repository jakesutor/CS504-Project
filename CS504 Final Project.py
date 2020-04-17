#!/usr/bin/env python
# coding: utf-8

# In[6]:


# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 21:20:05 2020

@author: jakes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn                   import metrics
from sklearn.preprocessing     import StandardScaler
from sklearn.model_selection   import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics           import accuracy_score, mean_squared_error, confusion_matrix
from sklearn.linear_model      import LogisticRegression
from sklearn.pipeline          import Pipeline
from sklearn.naive_bayes       import MultinomialNB
from sklearn.neighbors         import KNeighborsClassifier
from sklearn.tree              import DecisionTreeClassifier
from sklearn.ensemble          import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.svm               import SVC

stats = pd.read_csv(r'C:\Users\jakes\Documents\GitHub\CS504-Project\Season_Stats.csv')

# only grabbing years 2007-2018 (train on 2007-2017, test on 2018)
stats = stats[stats['Year'] > 2005].reset_index(drop=True) 

# dropping these two columns which are both entirely blank
stats = stats.drop(columns = ['blanl','blank2','Unnamed: 0'])


# In[7]:


stats.head()


# In[12]:


# group by year first, then player, then team
# this will let us see which players had multiple teams in a single year
# TOT value for teams is an aggregation of all the team data for the player
teams = stats.groupby(['Year','Player','Tm','ALL_STAR']).sum()
teams.head()


# In[13]:


# resetting the index will preserve the return order of the rows
# but the columns are treated a
teams = teams.reset_index()
teams.head(10)


# In[19]:


cols = teams.columns.values.tolist()


# In[15]:


# create a blank dataframe using the column values of Teams dataframe
# we are doing this so we can append values to it
TOT = pd.DataFrame(columns=cols)


# In[16]:


# this will give us all the rows where team == TOT
for i in range(len(teams)):
    if(teams['Tm'][i]=='TOT'):
        TOT.loc[i] = teams.loc[i]


# In[18]:


TOT.head()


# In[20]:


teams = teams.drop_duplicates(subset=['Player','Year'],keep=False)


# In[21]:


finalTeams = pd.concat([teams,TOT])
finalTeams = finalTeams.sort_values(by=['Year','Player'])
finalGrouped = finalTeams.groupby(['Year','Player','Tm','ALL_STAR']).sum()


# In[35]:


finalTeams = finalTeams.astype({'Year': 'int64', 'ALL_STAR':'bool','Age': 'int64','G':'int64','GS':'int64',
                                'MP':'int64','FG':'int64','FGA':'int64','3P':'int64','3PA':'int64',
                                '2P':'int64','2PA':'int64','FT':'int64','FTA':'int64','ORB':'int64',
                                'DRB':'int64','TRB':'int64','AST':'int64','STL':'int64','BLK':'int64',
                                'TOV':'int64','PF':'int64','PTS':'int64'})
finalTeams.dtypes
# Confirm that there are no negative values
finalTeams.describe()
finalTeams[(finalTeams.ix[:,5:47] < 0).all(1)]

# In[22]:


finalTeams[finalTeams['Player']=='Isaiah Thomas']


# In[23]:


finalTeams.to_csv('Final_Season_Stats.csv')


# # EDA/ QA
#code for indexing using rows: df.loc[]

#Define function to accept a df and value as arguement and return a list of index positions of all occurences
def getIndexes(dfObj, value):

    listOfPos = list()

    result = dfObj.isin([value])

    seriesObj = result.any()
    columnNames = list(seriesObj[seriesObj == True].index)

    for col in columnNames:
        rows = list(result[col][result[col] == True].index)
        for row in rows:
            listOfPos.append((row, col))

    return listOfPos

topOBPM = getIndexes(finalTeams, 47.8)
topWS = getIndexes(finalTeams, 20.3)
topeFG = getIndexes(finalTeams, 1.50000)
for i in range(len(topeFG)):
    print(i, topeFG[i])
    
# remove players where games < 10

teamsFiltered = finalTeams[finalTeams['G'] >= 10]

#remove players where minutes played <= 50

teamsFiltered = teamsFiltered[teamsFiltered['MP'] > 50]

teamsFiltered.describe() 

#create scatterplot for points scored and field goal %
plt.scatter(teamsFiltered['PTS'], teamsFiltered['FG%'])
plt.axhline(y=0.5, color='black')
plt.title('Field Goal Percentage vs Points Scored')

#create plot of All-star status and points scored
plt.scatter(teamsFiltered['PTS'], teamsFiltered['ALL_STAR'])
plt.title('All-Star Status vs Points Scored')

#all-star vs age
plt.scatter(teamsFiltered['Age'], teamsFiltered['ALL_STAR'])
plt.title('All-Star Status vs Age')

#all-star vs obpm
plt.scatter(teamsFiltered['OBPM'], teamsFiltered['ALL_STAR'])
plt.title('All-Star Status vs OBPM')

#all-star vs games
plt.scatter(teamsFiltered['G'], teamsFiltered['ALL_STAR'])
plt.title('All-Star Status vs Games Played')

#all-star vs TOV
plt.scatter(teamsFiltered['TOV'], teamsFiltered['ALL_STAR'])
plt.title('All-Star Status vs Turnovers')

#all-star vs WS
plt.scatter(teamsFiltered['WS/48'], teamsFiltered['ALL_STAR'])
plt.title('All-Star Status vs Win Share')

#all-star vs BPM
plt.scatter(teamsFiltered['BPM'], teamsFiltered['ALL_STAR'])
plt.title('All-Star Status vs BPM')

#all-star vs PER
plt.scatter(teamsFiltered['PER'], teamsFiltered['ALL_STAR'])
plt.title('All-Star Status vs Player Efficiency Rating')

#remove players where minutes played < 200
teamsFiltered2 = teamsFiltered[teamsFiltered['MP'] >= 200]


#running previous charts on MP >= 200 df  (only %based charts effected)
#all-star vs obpm
plt.scatter(teamsFiltered2['OBPM'], teamsFiltered2['ALL_STAR'])
plt.title('All-Star Status vs OBPM')
plt.text(-5, 0.5, 'Minimum 200 minutes played')


#all-star vs BPM
plt.scatter(teamsFiltered2['BPM'], teamsFiltered2['ALL_STAR'])
plt.title('All-Star Status vs BPM')
plt.text(-5, 0.5, 'Minimum 200 minutes played')

#all-star vs PER
plt.scatter(teamsFiltered2['PER'], teamsFiltered2['ALL_STAR'])
plt.title('All-Star Status vs Player Efficiency Rating')
plt.text(10, 0.5, 'Minimum 200 minutes played')

allstars = teamsFiltered2[teamsFiltered2['ALL_STAR']]

nonallstars = teamsFiltered2[teamsFiltered2['ALL_STAR']==False]


nonallstars['PTS'].describe()
allstars['PTS'].describe()

teamsFiltered2.dtypes


df = teamsFiltered2[teamsFiltered2['Year'] != 2017]
test_df = teamsFiltered2[teamsFiltered2['Year'] == 2017]
test_df.drop(test_df['ALL_STAR'])

df['player_index'] = df['Player'] + ': ' + df['Year'].astype(str)
df.set_index(df['player_index'], inplace = True)
df.drop(columns = ['player_index'], inplace = True)

drop = ['Player','Year','Tm','ALL_STAR','G','MP','GS','3PAr','FTr','WS/48','FG','FGA','3P','3PA','2P','2PA','FT','FTA']
features = df.drop(columns = drop).columns
x = df[features]
y = df['ALL_STAR']

print(x.shape)
y.shape


#### LOGISTIC REGRESSION ####
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid,cv=10)
logreg_cv.fit(x_train,y_train)
print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)

logreg2=LogisticRegression(C=1,penalty="l2")
model_res = logreg2.fit(x_train,y_train)

print("score",logreg2.score(x_test,y_test))


#Testing Logistic Regression
model_res.predict(x_train)
x_train['ALL_STAR_PROB'] = model_res.predict_proba(x_train)[:,1]


#adding target variable back in for evaluation
x_train['ALL_STAR'] = y_train

#creating column in X_test df with the predicted probability
x_test['ALL_STAR_PROB'] = model_res.predict_proba(x_test)[:, 1]

#adding target variable back in for evaluation
x_test['ALL_STAR'] = y_test

#Combining dataframes for evaluation
train_df = pd.concat([x_train, x_test])

#rounding values
train_df['ALL_STAR_PROB'] = np.round(train_df['ALL_STAR_PROB'], 2)

train_df['player_index'] = train_df.index
train_df[['Player','Year']] = train_df['player_index'].str.split(': ', expand = True)
train_df.drop(columns = ['player_index'], inplace = True)

#looking at top 10 most likely players to become an All-Star
train_df.sort_values(by = 'ALL_STAR_PROB', ascending = False).head(20)

test_df[features].head()

test_proba = test_df[features]


model_res.predict(test_proba)

model_res.predict_proba(test_proba)[:5]

all_star_proba = model_res.predict_proba(test_proba)[:,1]
all_star_proba[:15]

test_df['ALL_STAR_PROB'] = np.round(all_star_proba,2)

test_df.sort_values(by = 'ALL_STAR_PROB', ascending = False).head(20)


model_res.feature_importances_

best_features = pd.DataFrame(model_res.feature_importances_.reshape(1,32), columns = features).T
best_features.rename(columns = {0: 'feature_importance'}, inplace = True)
best_features.sort_values(by = 'feature_importance', ascending = False)



#### RANDOM FOREST TEST ####


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 42, stratify = y)

ss = StandardScaler()
ss.fit(x_train)
x_train_sc = ss.transform(x_train)
x_test_sc = ss.transform(x_test)

# Random Forest
rf = RandomForestClassifier(random_state = 42)

rf_params = {}

gs = GridSearchCV(rf, param_grid = rf_params, cv=3, verbose = 1)
gs.fit(x_train, y_train)

print('Unscaled')
print(f'CrossVal Score: {gs.best_score_}')
print(f'Training Score: {gs.score(x_train, y_train)}')
print(f'Testing Score: {gs.score(x_test, y_test)}')
print(gs.best_params_)


rf = RandomForestClassifier(random_state = 42)

rf_params = {'n_estimators': [50, 60, 70],
             'max_depth': [None, 50],
             'min_samples_split': [2, 3, 5, 7],
             'min_samples_leaf': [1, 2, 3, 4]}

rf_gs = GridSearchCV(rf, param_grid=rf_params, cv=3, verbose = 1)
rf_gs.fit(x_train, y_train)

print('Unscaled')
print(f'CrossVal Score: {rf_gs.best_score_}')
print(f'Training Score: {rf_gs.score(x_train, y_train)}')
print(f'Testing Score: {rf_gs.score(x_test, y_test)}')
print(rf_gs.best_params_)



rf_best = RandomForestClassifier(max_depth= None, 
                                 min_samples_leaf= 1, 
                                 min_samples_split= 5,
                                 n_estimators= 50,
                                 random_state= 42)

rf_best.fit(x_train, y_train)

print(f'CrossVal Score: {cross_val_score(rf_best, x_train, y_train).mean()}')
print(f'Training Score: {rf_best.score(x_train, y_train)}')
print(f'Testing Score: {rf_best.score(x_test, y_test)}')


rf_gs.predict_proba(x_test)[:10]

x_train['ALL_STAR_PROB'] = rf_best.predict_proba(x_train)[:,1]


#adding target variable back in for evaluation
x_train['ALL_STAR'] = y_train

#creating column in X_test df with the predicted probability
x_test['ALL_STAR_PROB'] = rf_best.predict_proba(x_test)[:, 1]

#adding target variable back in for evaluation
x_test['ALL_STAR'] = y_test

#Combining dataframes for evaluation
train_df = pd.concat([x_train, x_test])

#rounding values
train_df['ALL_STAR_PROB'] = np.round(train_df['ALL_STAR_PROB'], 2)

train_df['player_index'] = train_df.index
train_df[['Player','Year']] = train_df['player_index'].str.split(': ', expand = True)
train_df.drop(columns = ['player_index'], inplace = True)



#looking at top 10 most likely players to become an All-Star
train_df.sort_values(by = 'ALL_STAR_PROB', ascending = False).head(10)

test_df[features].head()

test_proba = test_df[features]


rf_best.predict(test_proba)

rf_best.predict_proba(test_proba)[:5]

all_star_proba = rf_best.predict_proba(test_proba)[:,1]
all_star_proba[:15]

test_df['ALL_STAR_PROB'] = np.round(all_star_proba,2)

test_df.sort_values(by = 'ALL_STAR_PROB', ascending = False).head(20)



rf_best.feature_importances_

best_features = pd.DataFrame(rf_best.feature_importances_.reshape(1,32), columns = features).T
best_features.rename(columns = {0: 'feature_importance'}, inplace = True)
best_features.sort_values(by = 'feature_importance', ascending = False)





