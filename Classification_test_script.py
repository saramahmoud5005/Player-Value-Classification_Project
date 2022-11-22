#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import  seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from scipy.stats import f_oneway
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
get_ipython().run_line_magic('pip', 'install --upgrade category_encoders')
from category_encoders import TargetEncoder
from sklearn import preprocessing
import joblib


# In[ ]:


player_data=pd.read_csv()


# **Drop columns**

# In[ ]:


#Drop columns
player_data.drop(['national_team','national_team_position','tags',
                  'club_team','club_position','traits','national_jersey_number','club_jersey_number','CAM',
                  'CB','CDM','CF','CM','RAM','RB','RCB','RCM','RDM','RF','RM','RS','RW','RWB','ST','GK_handling',
                  'GK_kicking','GK_positioning','GK_reflexes','agility','ball_control','curve','dribbling',
                  'freekick_accuracy','long_passing','long_shots','marking','penalties','positioning','reactions',
                  'release_clause_euro','short_passing','shot_power', 'sliding_tackle','sprint_speed','standing_tackle','volleys',
                  'id','name','full_name','birth_date','height_cm','nationality','positions'],axis=1, inplace=True)


# **Handle nulls**

# In[ ]:


#fill null with zero
player_data.fillna({'national_rating':0,'club_join_date':0,'contract_end_year':0},inplace=True)


###################
ModePerCategory=pd.read_csv("ModePerCategory.csv")
ModePerCategory.drop(['CAM','CB','CDM','CF','CM','RAM','RB','RCB','RCM','RDM','RF','RM','RS','RW','RWB','ST'],inplace=True)
def impute_missing_occ (row):
    index_no = ModePerCategory.columns.get_loc(column_name)
  
    if pd.isnull(row[column_name]) :
        if row[["PlayerLevel"]].values== 'A':
            return ModePerCategory.iloc[0,index_no]
        elif row[["PlayerLevel"]].values== "B":
            return ModePerCategory.iloc[1,index_no]  
        elif row[["PlayerLevel"]].values =="C":
            return ModePerCategory.iloc[2,index_no]  
        elif row[["PlayerLevel"]].values=="D":
            return ModePerCategory.iloc[3,index_no] 
        elif row[["PlayerLevel"]].values=="S":
            return ModePerCategory.iloc[4,index_no]    

    else:
        return row[[column_name]]



for i,column_name in enumerate(ModePerCategory.columns):
    player_data[column_name]=player_data.apply(impute_missing_occ,axis=1) 


# In[ ]:


#impute nulls
statistics=pd.read_csv('Statistics.csv')

for i,col in enumerate(player_data.columns):
  if player_data[col].isnull().sum() > 0:
    player_data[col].fillna(statistics[col],inplace=True)

print(player_data.isnull().sum())


# **Handle categories**

# In[ ]:


def body_type_encoding(row):
    if row[["body_type"]].values == "Lean" :
        return 1;
    elif row[["body_type"]].values == "Normal" :
        return 2;  
    elif row[["body_type"]].values == "Stocky" :
        return 3;     
    else:
        return 2;

player_data["body_type"]=player_data.apply(body_type_encoding,axis=1)  


#work rate encoding
def work_rate_encoding(row):
    if row[["work_rate"]].values == "Low/ Low" :
        return 1;
    elif row[["work_rate"]].values == "Low/ Medium" :
        return 2;  
    elif row[["work_rate"]].values == "Medium/ Medium" :
        return 3;  
    elif row[["work_rate"]].values == "Low/ High" :
        return 4;
    elif row[["work_rate"]].values == "Medium/ Low" :
        return 5;
    elif row[["work_rate"]].values == "Medium/ High" :
        return 6;
    elif row[["work_rate"]].values == "High/ Low" :
        return 7;
    elif row[["work_rate"]].values == "High/ Medium" :
        return 8;
    elif row[["work_rate"]].values == "High/ High" :
        return 9;                           
    else:
        return 3;

player_data["work_rate"]=player_data.apply(work_rate_encoding,axis=1)

#one hot encoding
from sklearn.preprocessing import OneHotEncoder
OHE=joblib.load("OHE")
preferred_foot_array=OHE.transform(player_data[['preferred_foot']]).toarray()
preferred_foot_labels = np.array(OHE.categories_).ravel()#to make it an array, and .ravel() to convert it from array of arrays to array of strings
preferred_foot = pd.DataFrame(preferred_foot_array, columns=preferred_foot_labels)
v=player_data['PlayerLevel']
player_data.drop(['PlayerLevel','preferred_foot'],axis=1,inplace=True)
player_data = pd.concat([player_data, preferred_foot], axis = 1)
player_data = pd.concat([player_data, v], axis = 1)



#target encoding
label_encoder=joblib.load('PlayerLevel_encoder')
player_data['PlayerLevel']=label_encoder.transform(player_data["PlayerLevel"])

positions_Cols = ['LWB','LW','LS','LM','LF','LDM','LCM','LCB','LB','LAM']
for column_name in positions_Cols:
    filename = column_name + "_class"  
    target_encoder=joblib.load(filename)
    player_data[column_name]=target_encoder.transform(player_data[column_name],player_data['PlayerLevel'])


# In[ ]:


player_data['club_join_date'] = player_data['club_join_date'].astype(str)
for i,cell in enumerate(player_data['club_join_date']):
      
      if cell=="0":
            player_data['club_join_date'][i]=int(0)
      else:
            player_data['club_join_date'][i]=int(cell.split('/')[2])    
player_data['club_join_date'] = player_data['club_join_date'].astype(int)
            


player_data['contract_end_year'] = player_data['contract_end_year'].astype(str)
for i,cell in enumerate(player_data['contract_end_year']):
      if cell=="0":
            player_data['contract_end_year'][i]=int(0)
      elif len(cell)>4:      
            player_data['contract_end_year'][i]=int("20"+cell.split('-')[2])
            
player_data['contract_end_year'] = player_data['contract_end_year'].astype(int)      


for i,cell in enumerate(player_data['contract_end_year']):
    if player_data['contract_end_year'][i]==0 and player_data['club_join_date'][i]>0:
         player_data['club_join_date'][i]=0
    elif player_data['contract_end_year'][i]>0 and player_data['club_join_date'][i]==0:
         player_data['contract_end_year'][i]=0

# subtract contract_end_year from club_join_date
years_player_club=player_data['contract_end_year']-player_data['club_join_date']
player_data.insert(20,'years_player_club',years_player_club)
# Drop contract_end_year & club_join_date
player_data.drop(['contract_end_year','club_join_date'], axis=1, inplace=True)


# **predict with best model**

# In[ ]:


Y=player_data['PlayerLevel']#label
X=player_data.drop(['PlayerLevel'],axis=1)#features


# In[ ]:


#feature scaling
scaler = joblib.load('MinMaxScaler')
X = pd.DataFrame(scaler.transform(X), columns=X.columns)
X.head()


# In[ ]:


#adaboost clasifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier



savedmodel =joblib.load("classification_test")

y_pred=savedmodel.predict(X)
accuracy = np.mean(y_pred == Y)
print("Accuracy of testing",accuracy)
test_err = metrics.mean_squared_error(Y, y_pred)
print('Test subset (MSE): ',test_err)

