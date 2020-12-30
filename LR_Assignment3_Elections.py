# -*- coding: utf-8 -*-
"""
Created on Thu May 14 22:24:59 2020

@author: RAVI
"""
import pandas as pd
import numpy as np

#Importing Data
Elections = pd.read_csv("C:/RAVI/Data science/Assignments/Module 9 Logistic regression/LR Assignment dataset3/election_data.csv/election_data.csv")
Elections.head()

Elections = Elections.drop( 0, axis=0)

Elections.columns="Electionid","Result","Year","Amountspent","Popularityrank"

# Imputating the missing values           
# Elections1 = Elections.apply(lambda x:x.fillna(x.value_counts().index[0]))

### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
train_data,test_data = train_test_split(Elections,test_size = 0.2) # 20% test data


# Model building 
import statsmodels.formula.api as sm
logit_model = sm.logit('Result ~  Electionid + Year + Amountspent + Popularityrank', data = train_data).fit()


#logit = sm.glm('Result ~  Electionid + Year + Amountspent + Popularityrank',data = train_data).fit() 
#logit.summary()
help(sm.logit)
help(sm.glm)
#summary
logit_model.summary()
#logit.summary()

predict = logit_model.predict(pd.DataFrame(test_data[['Electionid','Year','Amountspent','Popularityrank']]))

from sklearn.metrics import confusion_matrix

cnf_matrix = confusion_matrix(test_data['Result'], predict > 0.5 )
cnf_matrix

accuracy = (153+115)/(153+65+69+115)
accuracy


