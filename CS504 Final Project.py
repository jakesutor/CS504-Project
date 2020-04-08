# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 21:20:05 2020

@author: jakes
"""

import pandas as pd

stats = pd.read_csv('Seasons_Stats.csv')

# only grabbing years 2007-2018 (train on 2007-2017, test on 2018)
stats = stats[stats['Year'] > 2006].reset_index(drop=True) 

# dropping these two columns which are both entirely blank
stats = stats.drop(columns = ['blanl','blank2','Unnamed: 0'])

# group by year first, then player, then team
# this will let us see which players had multiple teams in a single year
# TOT value for teams is an aggregation of all the team data for the player
teams = stats.groupby(['Year','Player','Tm']).sum()
teams.head()

# resetting the index will preserve the return order of the rows
# but the columns are treated a
teams = teams.reset_index()
teams.head(10)

cols = teams.columns.values.tolist()

# create a blank dataframe using the column values of Teams dataframe
# we are doing this so we can append values to it
TOT = pd.DataFrame(columns=cols)

# this will give us all the rows where team == TOT
for i in range(len(teams)):
    if(teams['Tm'][i]=='TOT'):
        TOT.loc[i] = teams.loc[i]

        # now there are no duplicate values. 
# we only have the TOT values for each player in a given season
noDupes = teams[teams['Player'].isin(TOT['Player'])==False]

finalTeams = pd.concat([noDupes, TOT])
finalTeams = finalTeams.sort_values(by=['Year','Player'])

final = finalTeams.groupby(['Year','Player','Tm']).sum()
final.head()
finalTeams.head()
