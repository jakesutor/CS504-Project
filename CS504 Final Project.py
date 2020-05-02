# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 18:54:09 2020
@author: prahi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

import seaborn as sns

from sklearn                   import metrics, preprocessing
from sklearn.feature_selection import RFECV
from sklearn.model_selection   import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics           import accuracy_score, mean_squared_error, confusion_matrix, classification_report, precision_recall_curve, roc_curve
from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.preprocessing     import StandardScaler


stats = pd.read_csv(r'C:\Users\jakes\Documents\GitHub\CS504-Project\Season_Stats.csv')

# only grabbing years 2006-2017 (train on 2007-2016, test on 2017)
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


# In[22]:


finalTeams[finalTeams['Player']=='Isaiah Thomas']


# In[23]:


#finalTeams.to_csv('Final_Season_Stats.csv')


# # EDA/ QA


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

#testing outliers in data
topOBPM = getIndexes(finalTeams, 47.8)
topWS = getIndexes(finalTeams, 20.3)
topeFG = getIndexes(finalTeams, 1.50000)
for i in range(len(topeFG)):
    print(i, topeFG[i])

# remove outliers by removing players where games < 10
teamsFiltered = finalTeams[finalTeams['G'] >= 10]

#remove outliers by removing players where minutes played <= 50
teamsFiltered = teamsFiltered[teamsFiltered['MP'] > 50]
teamsFiltered.describe()

#plot different variables to better understand data
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


#further remove outliers by removing players where minutes played < 200
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


teamsFiltered2.dtypes
teamsFiltered2['PPG'] = teamsFiltered2['PTS']/teamsFiltered2['G']

#remove 2017 season from test set, and remove y-values to prepare for modeling
df = teamsFiltered2[teamsFiltered2['Year'] != 2017]
test_df = teamsFiltered2[teamsFiltered2['Year'] == 2017]
test_df.drop(test_df['ALL_STAR'])

#set player and year as index, removes them from variables to prepare for modeling
df['player_index'] = df['Player'] + ': ' + df['Year'].astype(str)
df.set_index(df['player_index'], inplace = True)
df.drop(columns = ['player_index'], inplace = True)

#remove redundant variables that may display collinearity
drop = ['Player','Year','Tm','ALL_STAR','G','MP','GS','3PAr','FTr','TRB%','WS','WS/48','BPM','FG','FGA','3P','3PA','2P','2PA','FT','FTA','ORB','DRB','TRB','AST','STL','BLK','TOV','PTS']
features = df.drop(columns = drop).columns
x = df[features]
y = df['ALL_STAR']

print(x.shape)
y.shape

#scale the data (which creates an array) and convert array to df, then QC
#x_scaled = StandardScaler().fit_transform(x.values)
#x_scaled_df = pd.DataFrame(x_scaled, index=x.index, columns=x.columns)
#x.shape
#x_scaled_df.shape

#### LOGISTIC REGRESSION ####
#create an instance of the Logistic regression function, and set max iterations to 200
#to allow for the model to converge
logreg=LogisticRegression(solver='liblinear',max_iter=200)

# Create the RFECV object in order to determine the variables to keep based on accuracy
rfecv = RFECV(estimator=logreg, step=1, scoring='accuracy',cv=3)
rfecv.fit(x, y)

print("Optimal number of features: %d" % rfecv.n_features_)
print('Selected features: %s' % list(x.columns[rfecv.support_]))

#plot the number of features vs. cross-validation scores
plt.figure(figsize=(10,6))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


#modify data to include only the optimal features
selected_features = list(x.columns[rfecv.support_])
#x_scaled_filtered = x_scaled_df[selected_features]
y = df['ALL_STAR']
x = df[selected_features]


#### RANDOM FOREST TEST ####

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 42, stratify = y)

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

# Create a column in the df with the predicted probability
x_train['ALL_STAR_PROB'] = rf_best.predict_proba(x_train)[:,1]

x_train['ALL_STAR'] = y_train

# Create a column in the df with the predicted probability
x_test['ALL_STAR_PROB'] = rf_best.predict_proba(x_test)[:, 1]

x_test['ALL_STAR'] = y_test

# Combine the training and testing datasets
train_df = pd.concat([x_train, x_test])
train_df['ALL_STAR_PROB'] = np.round(train_df['ALL_STAR_PROB'], 2)

train_df['player_index'] = train_df.index
train_df[['Player','Year']] = train_df['player_index'].str.split(': ', expand = True)
train_df.drop(columns = ['player_index'], inplace = True)

# Players most likely to become an All-Star in the training set
train_df.sort_values(by = 'ALL_STAR_PROB', ascending = False).head(10)

test_df[selected_features].head()
test_proba = test_df[selected_features]

rf_best.predict(test_proba)
rf_best.predict_proba(test_proba)[:5]

all_star_proba = rf_best.predict_proba(test_proba)[:,1]
all_star_proba[:15]

test_df['ALL_STAR_PROB'] = np.round(all_star_proba,2)

# Players most likely to become an All-Star in the testing set
test_df.sort_values(by = 'ALL_STAR_PROB', ascending = False).head(10)



best_features = pd.DataFrame(rf_best.feature_importances_.reshape(1,22), columns = selected_features).T
best_features.rename(columns = {0: 'feature_importance'}, inplace = True)
best_features = best_features.sort_values(by = 'feature_importance', ascending = False)


# Visualizations

# Top Features
plt.figure(figsize=(15,10))
plt.bar(best_features.index,best_features.feature_importance)
plt.title('Feature Importance', fontsize=20)



# LOGISTIC REGRESSION

top_features = ['PER', 'PPG', 'VORP', 'OWS', 'USG%', 'DWS', 'OBPM', 'AST%', 'ORB%']
y = df['ALL_STAR']
x = x[top_features]



#descriptive summary of scaled data, with p-values
logit_model=sm.Logit(y,x)
result=logit_model.fit()
print(result.summary2())

#split the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.3, random_state=0)

#create the model
logisreg = LogisticRegression()
model_res = logisreg.fit(x_train,y_train)
logisreg.fit(x_train,y_train)

#evaluate model with test split
y_pred = logisreg.predict(x_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logisreg.score(x_test, y_test)))

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
print(classification_report(y_test, y_pred))

#confusion matrix chart
group_names = ['True Neg','False Pos','False Neg', 'True Pos']
group_precisions = ['Precision = 0.97', '', '', 'Precision = 0.80']
group_counts = ["{0:0.0f}".format(value) for value in
                conf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     conf_matrix.flatten()/np.sum(conf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}\n{v4}" for v1, v2, v3, v4 in
          zip(group_names,group_counts,group_percentages, group_precisions)]
labels = np.asarray(labels).reshape(2,2)
sns.axes_style("dark")
sns.heatmap(conf_matrix, annot=labels, fmt='', linewidths=0.9, cmap='Blues', linecolor="dimgray", annot_kws={"size":12},
            robust=True)

#Testing Logistic Regression
model_res.predict(x_train)
x_train['ALL_STAR_PROB'] = model_res.predict_proba(x_train)[:,1]
x_train['ALL_STAR'] = y_train

# Predicted Probability of becoming an All-Star
x_test['ALL_STAR_PROB'] = model_res.predict_proba(x_test)[:, 1]

x_test['ALL_STAR'] = y_test

# Combine the training and testing data sets
train_df = pd.concat([x_train, x_test])


train_df['ALL_STAR_PROB'] = np.round(train_df['ALL_STAR_PROB'], 2)

train_df['player_index'] = train_df.index

train_df[['Player','Year']] = train_df['player_index'].str.split(': ', expand = True)

train_df.drop(columns = ['player_index'], inplace = True)

train_df.to_csv('Probability_DF_All_Star.csv')
# Players most likely to become an All-Star from training set
train_df.sort_values(by = 'ALL_STAR_PROB', ascending = False).head(10)

test_df[top_features].head()
test_proba = test_df[top_features]

model_res.predict(test_proba)
model_res.predict_proba(test_proba)[:5]

all_star_proba = model_res.predict_proba(test_proba)[:,1]
all_star_proba[:15]

test_df['ALL_STAR_PROB'] = np.round(all_star_proba,2)

# Players most likely to become an All-Star from testing set
test_df.sort_values(by = 'ALL_STAR_PROB', ascending = False).head(10)



# Visualizations

# Checking the top two features - clearly the top right are reserved for All-Stars
plt.figure(figsize=(15,10))
sns.lmplot('PER', 'VORP', teamsFiltered2, hue='ALL_STAR', fit_reg=False)
plt.title('PER Against VORP', fontsize=20)
plt.xlabel('PER', fontsize=16)
plt.ylabel('VORP', fontsize=16)
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.show()


# Checking PTS against MP - all-stars are slightly better but less clearly
plt.figure(figsize=(15,10))
sns.lmplot('PTS', 'MP', teamsFiltered2, hue='ALL_STAR', fit_reg=False)
plt.title('PTS Against MP', fontsize=20)
plt.xlabel('PTS', fontsize=16)
plt.ylabel('MP', fontsize=16)
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.show()
