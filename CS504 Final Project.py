#!/usr/bin/env python
# coding: utf-8

# In[6]:


# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 21:20:05 2020

@author: jakes
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

stats = pd.read_csv('Season_Stats.csv')

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

    
# there are no null values which is excellent
stats.isnull().sum()

# quick look at sumary stats for numeric cols
stats.describe()

# quick look at summary stats for categorical cols
stats.describe(include=['O'])

# create a copy of the all stars column but encode values with 1 or 0
stats['AllStarEncoded'] = stats['ALL_STAR'].replace({'No':0,'Yes':1})

# look at ALL the all star players and the number of all star appearances they have
# this only looks at players that have actually been an all star
# we can say = 0 in line 2 to make it only players who have not been in the game
allStars = stats[['Player','AllStarEncoded']].groupby('Player').sum().sort_values(by='AllStarEncoded', ascending = False).reset_index()
allStars = allStars[allStarGames['AllStarEncoded'] > 0]
allStars.head()

# looking at the distribution of the number of times a player appears in a game
# mostly people are only in one game and the numbers really fall off after 4 games
# lebron has most
sns.countplot(x = 'AllStarEncoded', data=allStars)

# this will give us the 10 features most correlated with all star status
# i am doing head on the top 11 but ignoring the first row because
# the first row is just all star status, which is 100% correlated with itself
statsCorr = stats.corr()
statsCorr = statsCorr[['AllStarEncoded']]
statsCorr['AllStarEncoded'].sort_values(ascending=False).head(11)[1:]

# again looking visually at features that are nicely correlated with all star status
sns.heatmap(data = statsCorr)

    
