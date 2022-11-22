#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
#%pip install --upgrade category_encoders
from category_encoders import TargetEncoder
from sklearn import preprocessing
import time


# **Load player player_data set & print head**

# In[ ]:


player_data=pd.read_csv("player-classification.csv")
player_data.head()


# **Display some info about player_data set**

# In[ ]:


player_data.info();


# In[ ]:


data=player_data.copy()
data.head()


# In[ ]:


#preprocessing of traits column
player_data['traits'] = player_data['traits'].fillna("Technical Dribbler (CPU AI Only)")# fill nulls with mod

traits_text=""
for i in player_data['traits']:
    traits_text=traits_text+','
    traits_text=traits_text+str(i)

trait_list = traits_text.split(',')

unique_list = []
for x in trait_list:
    if x not in unique_list:
        unique_list.append(x)

print(len(unique_list))


# **Check nulls**

# In[ ]:


f, ax = plt.subplots(figsize=(50, 6))
sns.heatmap(player_data.isnull(),yticklabels=False,cbar=False,cmap="viridis",ax=ax)


# **Handle Nulls**

# In[ ]:


#Drop columns
data.drop(['national_team','national_team_position','tags',
                  'club_team','club_position','traits','national_jersey_number','club_jersey_number'],axis=1, inplace=True)
#fill null with zero
data.fillna({'national_rating':0,'club_join_date':0,'contract_end_year':0},inplace=True) 
player_data.fillna({'national_rating':0,'club_join_date':0,'contract_end_year':0},inplace=True) 

#replace nulls with mode

data['wage'].fillna(player_data.wage.mode()[0],inplace=True)
data['release_clause_euro'].fillna(player_data.release_clause_euro.mode()[0],inplace=True)#save

player_data['wage'].fillna(player_data.wage.mode()[0],inplace=True)
player_data['release_clause_euro'].fillna(player_data.release_clause_euro.mode()[0],inplace=True)
#replace nulls with mean
data['club_rating'].fillna(int(data['club_rating'].mean()),inplace=True)#save
player_data['club_rating'].fillna(int(data['club_rating'].mean()),inplace=True)


# replace nulls with mode per each category 

# In[ ]:


#get mode of each category(A,B,C,D,S) per column
df=player_data.iloc[:,65:92]
ModePerCategory=df.groupby('PlayerLevel').agg(lambda x: x.value_counts().index[0])
ModePerCategory


# In[ ]:


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

dd=df.drop(['PlayerLevel'],axis=1)

for i,column_name in enumerate(dd.columns):
    data[column_name]=data.apply(impute_missing_occ,axis=1)        


# In[ ]:


f, ax = plt.subplots(figsize=(50, 6))
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap="viridis",ax=ax)


# **Correlation**
# 
# 
# 1.   categorical with categorical
# 2.   numerical with categorical
# 
# 

# In[ ]:


get_ipython().run_line_magic('pip', 'install researchpy')
import researchpy as rp

catCols = player_data.select_dtypes("object").columns
V=np.zeros([df.shape[1],df.shape[1]])
for i,column_name in enumerate(df.columns):
    v=[]
    for j,name in enumerate(df.columns): 
        ctab,chi_statistic,expected=rp.crosstab(player_data[name],player_data[column_name],margins=False,test='chi-square',expected_freqs=True)
        v.append(chi_statistic.iloc[2,1])
    V[i]=v 

categories_corr=pd.DataFrame(V,index=df.columns,columns=df.columns)   
categories_corr 


# In[ ]:


for i,column_name in enumerate(player_data.columns):
    
    if i == 65:
        break
    if player_data.dtypes[column_name] == "object":
        continue
    CategoryGroupLists=player_data.groupby('PlayerLevel')[column_name].apply(list)
  # when P-Value > 0.05 that means columns are not correlated --> reject the column
    AnovaResults = f_oneway(*CategoryGroupLists)
    if AnovaResults[1] > 0.05:
        print(column_name, ' --> P-Value for Anova is: ', AnovaResults[1]) # --> columns to be dropped + (id)
        

corrr_categ_and_y = []

for i,column_name in enumerate(player_data.columns):
    if i == 65:
        break
    if player_data.dtypes[column_name] != "object":
        continue
    ctab,chi_statistic,expected=rp.crosstab(player_data[column_name],player_data["PlayerLevel"],margins=False,test='chi-square',expected_freqs=True)
    corrr_categ_and_y.append(chi_statistic.iloc[2,1]) 
    if corrr_categ_and_y[-1] < 0.1:
        print(column_name , "-->" , corrr_categ_and_y[-1]) # --> columns to be dropped +  (name & full name & nationality)        
        


# **Handling categories**
# 
# 1.   work_rate,body_type ==> label_encoding
# 2.  preffered_foot ==> one hot encoding
# 
# 1.   positions ==> split by (,),then apply labe encoding
# 
# 2.   last 27 columns ==> target encoding

# In[ ]:


# Splitting strings
data['club_join_date'] = data['club_join_date'].astype(str)
for i,cell in enumerate(data['club_join_date']):
      
        if cell=="0":
            data['club_join_date'][i]=int(0)
        else:
            data['club_join_date'][i]=int(cell.split('/')[2])    
data['club_join_date'] = data['club_join_date'].astype(int)
            


data['contract_end_year'] = data['contract_end_year'].astype(str)
for i,cell in enumerate(data['contract_end_year']):
        if cell=="0":
            data['contract_end_year'][i]=int(0)
        elif len(cell)>4:      
            data['contract_end_year'][i]=int("20"+cell.split('-')[2])
            
data['contract_end_year'] = data['contract_end_year'].astype(int)      


for i,cell in enumerate(data['contract_end_year']):
    if data['contract_end_year'][i]==0 and data['club_join_date'][i]>0:
         data['club_join_date'][i]=0
    elif data['contract_end_year'][i]>0 and data['club_join_date'][i]==0:
         data['contract_end_year'][i]=0

# subtract contract_end_year from club_join_date
years_player_club=data['contract_end_year']-data['club_join_date']
data.insert(20,'years_player_club',years_player_club)
# Drop contract_end_year & club_join_date
data.drop(['contract_end_year','club_join_date'], axis=1, inplace=True)


# In[ ]:


#handle position column
split_positions = data['positions'].str.split(',', expand = True).rename(columns = {0:"first_positions",1:"second_positions",2:"third_positions",3:"fourth_positions",})
split_positions = split_positions.fillna("0")  


ctab,chi_statistic,expected=rp.crosstab(split_positions['fourth_positions'],player_data['PlayerLevel'],margins=False,test='chi-square',expected_freqs=True)
print(chi_statistic)
#labelencoder = LabelEncoder()

# split_positions['label_first_pos'] =  labelencoder.fit_transform(split_positions['first_positions'])
# split_positions['label_second_pos'] =  labelencoder.fit_transform(split_positions['second_positions'])
# split_positions['label_third_pos'] =  labelencoder.fit_transform(split_positions['third_positions'])
# split_positions['label_fourth_pos'] =  labelencoder.fit_transform(split_positions['fourth_positions'])

# split_positions.drop('first_positions', axis=1, inplace=True)
# split_positions.drop('second_positions', axis=1, inplace=True)
# split_positions.drop('third_positions', axis=1, inplace=True)
# split_positions.drop('fourth_positions', axis=1, inplace=True)

# for i,c in enumerate(split_positions.columns):
#     data.insert(7+i,c,split_positions[c])

data.drop(['positions'],axis=1,inplace=True)


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

data["body_type"]=data.apply(body_type_encoding,axis=1)  


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

data["work_rate"]=data.apply(work_rate_encoding,axis=1)


#one hot encoding
data['preferred_foot'].unique()
one_hot_encoder = OneHotEncoder()
preferred_foot_array = one_hot_encoder.fit_transform(player_data[['preferred_foot']]).toarray()
preferred_foot_labels = np.array(one_hot_encoder.categories_).ravel()#to make it an array, and .ravel() to convert it from array of arrays to array of strings
preferred_foot = pd.DataFrame(preferred_foot_array, columns=preferred_foot_labels)
v=data['PlayerLevel']
data.drop(['PlayerLevel','preferred_foot'],axis=1,inplace=True)
data = pd.concat([data, preferred_foot], axis = 1)
data = pd.concat([data, v], axis = 1)





# In[ ]:


#target encoding 
import joblib
y_encoder=preprocessing.LabelEncoder()#save

data['PlayerLevel']=y_encoder.fit_transform(data["PlayerLevel"])

X=player_data.iloc[:,65:91]


for i,c in enumerate(X.columns):
    encoder=TargetEncoder()#save
    data[c]=encoder.fit_transform(data[c],data['PlayerLevel'])
  
    
    

data.info()


# **Drop dependent features**

# In[ ]:


#with the following function we can select highly correlated features
#it will remove the first feature that is correlated with anything other features
def correlation(corr_matrix,threshold):
    col_corr=set()
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j])>=threshold:
                colname=corr_matrix.columns[i] #getting the name of the column
                col_corr.add(colname)
    return col_corr 


# In[ ]:


feature_matrix=player_data.drop(['PlayerLevel'],axis=1)
corr_matrix=feature_matrix.corr()


# In[ ]:


corr_categories_features=correlation(categories_corr,1.0)
print(len(set(corr_categories_features)))
corr_categories_features


# In[ ]:


corr_features=correlation(corr_matrix,0.8)
print(len(set(corr_features)))
corr_features


# In[ ]:


df=data.copy()
#data=df
data.drop(['id','name','full_name','birth_date','height_cm','nationality'],axis=1,inplace=True)
data.drop(corr_features,axis=1,inplace=True)
data.drop(corr_categories_features,axis=1,inplace=True)
print(data.shape)


# In[ ]:


X=data.iloc[:,0:len(data.iloc[0,:])-1]#features
Y=data.iloc[:,-1]#label


# In[ ]:


#feature scaling
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
joblib.dump(scaler,"c_scaler")
X.head()


# In[ ]:


#split data int train 80%  test 20%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20,shuffle=True,random_state=10)

print(X_train.shape)
print(X_test.shape)


# Apply 
# 
# 1.   Decision tree(can handle categorical & numerical)
# 2.   Svm (can handle  numerical only)
# 
# 1.   Logestic regression(can handle  numerical only)
# 
# 
# 
# 
# 

# In[ ]:


#Dectionary to add in it mse of each model (key:name of model,value:mse),to use it in plotting graph
MSE={"Logestic":0,"SVM":0,'AdaBoost':0}
Training_time={"Logestic":0,"SVM":0,'AdaBoost':0}
Testing_time={"Logestic":0,"SVM":0,'AdaBoost':0}


# In[ ]:


#SVM
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
start_training=time.time()
params_grid = [
                {'kernel': ['rbf','linear'], 
                'gamma': [1e-3, 1e-4,0.1],
                'C': [1, 10, 100, 1000],
                'decision_function_shape':['ovo','ovr']
                }
              ]
svm_model = GridSearchCV(SVC(), params_grid, cv=5)
svm_model.fit(X_train, y_train)


# View the best parameters for the model found using grid search
print(svm_model.best_estimator_) 

final_model = svm_model.best_estimator_
end_training=time.time()
Training_time['SVM']=(end_training - start_training)
start_testing=time.time()
test_pred = final_model.predict(X_test)
test_err = metrics.mean_squared_error(y_test, test_pred)
end_testing=time.time()
Testing_time['SVM']=(end_testing - start_testing)
print('Best score for training data:', svm_model.best_score_,"\n") 
print("Training set score for SVM: %f" % final_model.score(X_train , y_train))
print("Testing  set score for SVM: %f" % final_model.score(X_test  , y_test ))


MSE["SVM"]=test_err
print("Test MSE:",test_err)


# In[ ]:


svm_model.best_estimator_.kernel


# In[ ]:


#Logestic regression
from sklearn.linear_model import LogisticRegression

start_training=time.time()
logModel = LogisticRegression(multi_class='multinomial')

param_grid = [    
    {
     'C' : [1,2,3,50,100,200],
     'max_iter' : [100, 1000,2500],
     
    }
]



clf = GridSearchCV(logModel, param_grid = param_grid, cv = 3, verbose=True, n_jobs=-1)
best_clf = clf.fit(X_train,y_train)

print(best_clf.best_estimator_)
print("Accuracy of training",best_clf.best_score_)
end_training=time.time()
Training_time['Logestic']=(end_training - start_training)
start_testing=time.time()
y_predection=best_clf.predict(X_test)
accuracy = np.mean(y_predection == y_test)
print("Accuracy of testing",accuracy)
end_testing=time.time()
Testing_time['Logestic']=(end_testing - start_testing)

test_err = metrics.mean_squared_error(y_test, y_predection)
MSE['Logestic']=test_err
print("Test MSE:",test_err)


# In[ ]:


#adaboost clasifier with depth 4
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4), algorithm="SAMME")
param_grid = [    
    {
        'n_estimators':[100,200,300,500],
     
    }
]



clf_bdt = GridSearchCV(bdt, param_grid = param_grid, cv = 3, verbose=3, n_jobs=-1)
ADa_clf = clf_bdt.fit(X_train,y_train)
print(ADa_clf.best_estimator_)
print("Average score",ADa_clf.best_score_)


y_pred=ADa_clf.predict(X_test)
y_train_predict=ADa_clf.predict(X_train)
test_accuracy = np.mean(y_pred == y_test)
train_accuracy=np.mean(y_train_predict == y_train)
print("Accuracy of training set with depth 4 : ",train_accuracy)
print("Accuracy of testing with depth 4 :  ",test_accuracy)
test_err = metrics.mean_squared_error(y_test, y_pred)
#MSE['AdaBoost']=test_err
#print('Test subset (MSE): ',test_err)


# In[ ]:


#Adaboost with depth 5
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), algorithm="SAMME")
param_grid = [    
    {
        'n_estimators':[100,200,300,500],
     
    }
]



clf_bdt = GridSearchCV(bdt, param_grid = param_grid, cv = 3, verbose=3, n_jobs=-1)
ADa_clf = clf_bdt.fit(X_train,y_train)
print(ADa_clf.best_estimator_)
print("Average score",ADa_clf.best_score_)

y_pred=ADa_clf.predict(X_test)
y_train_predict=ADa_clf.predict(X_train)
test_accuracy = np.mean(y_pred == y_test)
train_accuracy=np.mean(y_train_predict == y_train)
print("Accuracy of training set with depth 5 :",train_accuracy)
print("Accuracy of testing with depth 5 :",test_accuracy)
test_err = metrics.mean_squared_error(y_test, y_pred)


# In[ ]:


#Adaboost with depth 6
start_training=time.time()
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=6), algorithm="SAMME")
param_grid = [    
    {
        'n_estimators':[100,200,300,500],
     
    }
]



clf_bdt = GridSearchCV(bdt, param_grid = param_grid, cv = 3, verbose=3, n_jobs=-1)
ADa_clf = clf_bdt.fit(X_train,y_train)
print(ADa_clf.best_estimator_)
print("Average score",ADa_clf.best_score_)
end_training=time.time()
Training_time['AdaBoost']=(end_training - start_training)
start_testing=time.time()
y_pred=ADa_clf.predict(X_test)
y_train_predict=ADa_clf.predict(X_train)
test_accuracy = np.mean(y_pred == y_test)
train_accuracy=np.mean(y_train_predict == y_train)
end_testing=time.time()
Testing_time['AdaBoost']=(end_testing - start_testing)
print("Accuracy of training set with depth 6 : ",train_accuracy)
print("Accuracy of testing with depth 6 :",test_accuracy)
test_err = metrics.mean_squared_error(y_test, y_pred)
MSE['AdaBoost']=test_err


# bar graph (mse of all models)

# In[ ]:


plt.figure(figsize=(10, 6))
plt.title ("MSE of models with best param ")
y=list(MSE.values())
x=list(MSE.keys())
sns.barplot(x,y)
plt.show()


# In[ ]:


plt.figure(figsize=(10, 6))
plt.title ("Training time of models using GridSearchCV ")
y=list(Training_time.values())
x=list(Training_time.keys())
sns.barplot(x,y)
plt.show()


# In[ ]:


plt.figure(figsize=(10, 6))
plt.title ("Testing time of models ")
y=list(Testing_time.values())
x=list(Testing_time.keys())
sns.barplot(x,y)
plt.show()


# In[ ]:


#%pip install --default-timeout=100 xgboost
from xgboost import XGBClassifier
xgb = XGBClassifier()
start_training=time.time()
xgb.fit(X_train, y_train)
end_training=time.time()
Training_time.update(xgboost =(end_training - start_training)) 
start_testing=time.time()
y_train_predicted = xgb.predict(X_train)
prediction = xgb.predict(X_test)
end_testing=time.time()
Testing_time.update(xgboost =(end_testing - start_testing))
train_err = metrics.accuracy_score(y_train, y_train_predicted)
test_err = metrics.accuracy_score(y_test, prediction)
    
print('Train subset accuracy ', train_err)
print('Test subset accuracy' ,test_err)
test_err = metrics.mean_squared_error(y_test, prediction)
MSE.update(xgboost =test_err) 
print('Test subset (MSE): ',test_err)


# In[ ]:


joblib.dump(ADa_clf,"ADa_test")


# In[ ]:


#! pip install scikit-optimize
from skopt import BayesSearchCV
opt = BayesSearchCV(
    XGBClassifier(),
    {
      "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
      "max_depth"        : [ 3, 4, 5, 6, 8, 10,12],
      "min_child_weight" : [ 1, 3, 5, 7 ],
      "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
      "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    },
    n_iter=20,
    cv=3
)

opt.fit(X_train, y_train)

print("val. score: %s" % opt.best_score_)
print("test score: %s" % opt.score(X_test, y_test))


# In[ ]:


print("best params: %s" % str(opt.best_params_))

