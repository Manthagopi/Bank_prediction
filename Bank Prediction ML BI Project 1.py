#!/usr/bin/env python
# coding: utf-8

# In[132]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


# In[34]:


data = pd.read_csv('bank-marketing csv.csv')
data


# In[35]:


data.info()


# In[36]:


data.isnull().sum()


# There are No Null Values

# In[39]:


data.shape


# In[44]:


data.describe()


# In[45]:


data.head()


# Describe the Pdays column ,Make note of the mean,median and minnimum Values.Anything fishy in the values

# In[47]:


print("Mean of the pdays column is", data['pdays'].mean())
print("Median of the pdays column is", data['pdays'].median())
print("Min of the pdays column is", data['pdays'].min())


# If value is -1 i.e. It is an outlier because no. of days passed can't be negative.

# Describe the pdays column again, this time limiting yourself to the relevant values of pdays. How different are the mean and the median values?

# In[49]:


print("Mean of pdays column after eliminating -1 values is", data[data['pdays'] != -1]['pdays'].mean() )
print("Median of pdays column after eliminating -1 values is", data[data['pdays'] != -1]['pdays'].median() )


# After skipping the Pdays  "-1" outlier the mean and median values are changed to extent

# Plot a horizontal bar graph with the median values of balance for each education level value. 
# Which group has the highest median?

# In[77]:


data.groupby('education')['balance'].median().plot.barh(color='yellow')
plt.title('Education wise Median of Balance');
plt.ylabel('Education')
plt.xlabel('Balance');


# Tertiary Education Has the highest balance

# Make a box plot for pdays. Do you see any outliers?

# In[80]:


sns.boxplot(data.pdays, orient='H')


# Yes,there are lot of outliers in Pdays Column

# Coverting the categorical target Variable into Numerical Varible 

# In[81]:


data.response.replace({'no':0,'yes':1},inplace=True)


# In[91]:


data.response.sample(10)


# In[106]:


data.corr()


# In[108]:


plt.figure(figsize=(12,10))
sns.heatmap(data.corr(),annot=True);


# In[114]:


sns.countplot(data['education'],hue=data['response']);


# In[125]:


sns.boxplot(data['education'],data['age'],hue=data['response']);


# Are pdays and poutcome associated with the target  ?

# In[144]:


sns.pairplot(data,hue='response',diag_kws={'bw':2});


# In[145]:


sns.countplot(x='response',data=data);


# In[156]:


sns.countplot(y='job',data=data);


# In[152]:


sns.countplot(x='marital',data=data)


# In[150]:


sns.countplot(x='education',data=data)


# Are the features about the previous campaign data useful?

# In[170]:


data.columns


# In[166]:


categorical_columns = [column for column in data.columns[:-1] if data[column].dtype == 'O']
print(categorical_columns)
     


# In[167]:


for column in categorical_columns:
    print('Unique values in',column,'are',data[column].unique())


# In[175]:


for column in categorical_columns:
    pd.crosstab(data[column],data['response']).plot.bar()
    plt.title(column)


# 'poutcome' column is not assosciated with target column because it has more than 80% missing values.

# In[177]:


data.drop('poutcome',axis=1,inplace=True)


# In[179]:


data['pdays'].value_counts()


# In[182]:


data['previous'].value_counts().head(9)


# how do you handle the pdays column with a value of -1 where the previous campaign data is missing?

# In[183]:


data['pdays_no_contact'] = np.where(data['pdays']== -1,1,0)
data['pdays_no_contact'].value_counts()


# In[184]:


data.head()


# We created a new column since majority of users were not previously contacted. We are capturing importance of missing values.

# Handling Missing Values in Categorical columns

# In[185]:


for column in categorical_columns[:-1]:
    print(data[column].value_counts(),"\n")


# Handling Outliers in the Data.

# In[186]:


num_columns = [col for col in data.columns if col not in categorical_columns]
print(num_columns)


# In[189]:


dist=data.hist(figsize=(12,10)) # display numerical feature distribution


# In[191]:


##### Assuming Age follows A Gaussian Distribution we will calculate the boundaries which differentiates the outliers

upper_boundary = data['age'].mean() + 3* data['age'].std()
lower_boundary = data['age'].mean() - 3* data['age'].std()
print(lower_boundary), print(upper_boundary),print(data['age'].mean())


# In[220]:


index = data[(data['age']>upper_boundary) | (data['age']<lower_boundary)].index
data.drop(index=index,axis=0,inplace=True)


# In[193]:


data[(data['age']>upper_boundary) | (data['age']<lower_boundary)]


# In[194]:


##### Assuming Balance follows A Gaussian Distribution we will calculate the boundaries which differentiates the outliers

#### Lets compute the Interquantile range to calculate the boundaries
IQR=data.balance.quantile(0.75)-data.balance.quantile(0.25)

lower_bridge = data['balance'].quantile(0.25)-(IQR*1.5)
upper_bridge = data['balance'].quantile(0.75)+(IQR*1.5)
print(lower_bridge)
print(upper_bridge)


# In[195]:


data[(data['balance']>upper_bridge) | (data['balance']<lower_bridge)]


# In[196]:


index = data[(data['balance']>upper_bridge) | (data['balance']<lower_bridge)].index
data.drop(index=index,axis=0,inplace=True)


# In[197]:


data[(data['balance']>upper_bridge) | (data['balance']<lower_bridge)]


# In[ ]:


data.reset_index(inplace=True)


# In[200]:


data['balance'].hist(bins=20)


# In[203]:


data['age'].hist(bins=10)


# Outliers are handled now

# Handling Categorical columns.

# In[206]:


data.head()


# In[208]:


data['month'].unique()


# In[210]:


dictionary={'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12
}

data['month']=data['month'].map(dictionary)


# In[211]:


data['month'].unique()


# In[212]:


data.head()


# In[213]:


data1=data.copy()
data1.head()


# In[214]:


data1 = pd.get_dummies(data1,drop_first=True)
data1.head()


# In[215]:


data1.shape


# In[221]:


data1.loc[data1['pdays']==-1,'pdays']=0
data1['pdays'].head()


# # Feature Selection

# In[ ]:


# from sklearn.ensemble import ExtraTreesClassifier
# import matplotlib.pyplot as plt
# model=ExtraTreesClassifier()
# model.fit(X,y)

# print(model.feature_importances_)
# plt.figure(figsize=(10,10))
# ranked_features=pd.Series(model.feature_importances_,index=X.columns)
# ranked_features.nlargest(32).plot(kind='barh');


# # Handling Imbalanced Dataset

# In[233]:


#Check the percentage of 0 to 1
No_sub = len(data[data['response'] == 0])
Sub = len(data[data['response'] == 1])
percent_No_sub = (No_sub/len(data['response'])) * 100
percent_sub = (Sub/len(data['response'])) * 100

print('Percentage of subsription : ',percent_sub)
print('Percentage of no subscription : ', percent_No_sub)
print(No_sub)
print(Sub)
data['response'].value_counts().plot.bar();


# In[228]:


X = data1.drop('response',axis=1)
y = data1['response']


# In[231]:


get_ipython().system('pip install imbalanced-learn')


# In[239]:


from imblearn.combine import SMOTETomek
from collections import Counter

os=SMOTETomek(1)
X_ns,y_ns = os.fit_resample(X,y)
print("The number of classes before fit {}".format(Counter(y)))
print("The number of classes after fit {}".format(Counter(y_ns)))


# In[240]:


y_ns.value_counts()


# In[242]:


y_ns.head()


# In[248]:


X_ns.columns


# In[245]:


data['job'].unique()


# In[251]:


job = 'job_'+data['job'].unique()
job


# In[254]:


X_ns.drop(['job_management', 'job_technician', 'job_entrepreneur',
       'job_blue-collar', 'job_unknown', 'job_retired',
       'job_services', 'job_self-employed', 'job_unemployed',
       'job_housemaid', 'job_student'],axis=1,inplace=True)


# In[257]:


'marital_'+data['marital'].unique()


# In[258]:


X_ns.drop(['marital_married', 'marital_single'],axis=1,inplace=True)


# In[259]:


X_ns.drop(['targeted_yes', 'default_yes'],axis=1,inplace=True)


# In[260]:


X_ns.head()


# In[261]:


X_ns.shape


# In[263]:


y_ns.shape


# # Feature Scaling

# In[264]:


#### standarisation: We use the Standardscaler from sklearn library
from sklearn.preprocessing import StandardScaler


# In[265]:


scaler=StandardScaler()
### fit vs fit_transform
scaler.fit_transform(X_ns)


# In[266]:


X_scaled = pd.DataFrame(scaler.fit_transform(X_ns),columns=X_ns.columns)
X_scaled.head()


# # Model Development

# In[267]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X_scaled, y_ns, test_size=0.3, random_state=0)


# In[268]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)


# In[269]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[271]:


print(confusion_matrix(y_test,y_pred))


# In[272]:


print(accuracy_score(y_test,y_pred))


# In[273]:


print(classification_report(y_test,y_pred))


# # Cross Validation

# In[274]:


from sklearn.model_selection import cross_val_score
score=cross_val_score(lr,X_scaled,y_ns,cv=15)
score


# In[275]:


score.mean()


# # Random Forest Classifier

# In[276]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=20, random_state=0)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)


# In[277]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[278]:


print(confusion_matrix(y_test,y_pred))


# In[279]:


print(classification_report(y_test,y_pred))


# In[280]:


print(accuracy_score(y_test, y_pred))


# # Hyperparameter Optimization

# In[281]:


import numpy as np
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 50, num = 5)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt','log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 1000,15)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10,14]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4,6,8]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
              'criterion':['entropy','gini']}
print(random_grid)


# In[282]:


clf_randomcv = RandomizedSearchCV(estimator=clf,param_distributions=random_grid,n_iter=100,cv=3,verbose=2,
                               random_state=100,n_jobs=-1)
### fit the randomized model
clf_randomcv.fit(X_train,y_train)


# In[283]:


clf_randomcv.best_estimator_


# In[284]:


clf_randomcv.best_score_


# In[285]:


clf_randomcv.best_params_


# In[286]:


clf_best_random = clf_randomcv.best_estimator_


# In[287]:


y_pred = clf_best_random.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  
print(accuracy_score(y_test, y_pred))


# # K-Fold Cross Validation

# In[288]:


from sklearn.model_selection import cross_val_score
score=cross_val_score(clf_best_random,X_scaled,y_ns,cv=15)

score


# In[289]:


score.mean()


# Random Forest Performs much better as its average accuracy score is 87.3% to that of Logistic regression which has an accuracy of 85.8%.
# 
# I have used Accuracy as a metric to compare because I have handled the imbalanced data, would it be imbalanced I should have used F1-score.
