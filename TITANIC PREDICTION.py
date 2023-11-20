#!/usr/bin/env python
# coding: utf-8

# In[136]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import style
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC , LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , r2_score


# In[137]:


df = pd.read_csv(r"C:\Users\hplap\Downloads\archive (17)\Titanic-Dataset.csv")


# In[138]:


df.head()


# In[139]:


df.info()


# In[140]:


df.describe()


# In[141]:


df.isnull().sum()


# In[142]:


df.head()


# In[143]:


total = df.isnull().sum().sort_values(ascending = False)
percent_1  = df.isnull().sum()/df.isnull().count()*100
percent_2 = (round(percent_1,1).sort_values(ascending = False))
missing_data = pd.concat([total , percent_2],axis = 1 , keys = ['Total', '%'])
missing_data.head(5)


# In[144]:


df.columns.value_counts


# In[146]:


df['Fare'] = df['Fare'].astype(int)
df['Fare'] = df['Fare'].fillna(0)


# In[147]:


survived = 'survived'
not_survived = 'not survived'

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

women = df[df['Sex'] == 'female']
men = df[df['Sex'] == 'male']

ax = sns.distplot(women[women['Survived'] == 1].Age.dropna(), bins=18, label=survived, ax=axes[0], kde=False)
ax = sns.distplot(women[women['Survived'] == 0].Age.dropna(), bins=40, label=not_survived, ax=axes[0], kde=False)
ax.legend()
ax.set_title("Female")

ax = sns.distplot(men[men['Survived'] == 1].Age.dropna(), bins=18, label=survived, ax=axes[1], kde=False)
ax = sns.distplot(men[men['Survived'] == 0].Age.dropna(), bins=40, label=not_survived, ax=axes[1], kde=False)
ax.legend()
ax.set_title("Male")

plt.show()


# In[148]:


FacetGrid = sns.FacetGrid(df, row='Embarked', height=2.5, aspect=1.6)
FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None, order=None, hue_order=None)
FacetGrid.add_legend()

plt.show()


# In[149]:


sns.barplot(x = 'Pclass',y = 'Survived' , data = df)


# In[150]:


data = [df]
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0 , 'not_alone'] = 0
    dataset.loc[dataset['relatives'] == 0 , 'not_alone'] = 1
df['not_alone'].value_counts()


# In[151]:


df = df.drop(['PassengerId'],axis = 1)


# In[152]:


df.head()


# In[153]:


import re

deck_mapping = {'A': 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}

for dataset in [df]:  # Assuming df is the DataFrame you want to modify
    dataset['Cabin'] = dataset['Cabin'].fillna("U0")
    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    dataset['Deck'] = dataset['Deck'].map(deck_mapping)
    dataset['Deck'] = dataset['Deck'].fillna(0)
    dataset['Deck'] = dataset['Deck'].astype(int)

df = df.drop(['Cabin'], axis=1)


# In[154]:


df.info()


# In[155]:


data = [df]
for dataset in data:
    mean = df['Age'].mean()
    std = df['Age'].std()
    is_null = df['Age'].isnull().sum()
    #Compute random numbers 
    rand_age = np.random.randint(mean-std, mean+std, size = is_null)
    age_slice = df['Age'].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    df['Age'] = age_slice
    df['Age'] = df['Age'].astype(int)
df['Age'].isnull().sum()


# In[156]:


df['Embarked'].describe()


# In[157]:


common_value = 'S'
df = [df]
for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)


# In[158]:


data = df
title = {'Mr':1,'Miss': 2, 'Mrs' :3 , 'Master': 4, 'Rare': 5}

for dataset in data:
    dataset['Title'] = dataset['Name'].apply(lambda x: re.search(' ([A-Za-z]+)\.', x).group(1) if pd.notnull(x) else '')
    
    dataset['Title'] = dataset['Title'].replace(['Lady','Countless','Capt','Col','Don','Dr',\
                                                'Major','Rev','Jonkheer','Dona'],'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle','Miss')
    dataset['Title'] = dataset['Title'].replace('Ms','Miss')
    dataset['Title'] = dataset['Title'].replace('Mme','Mrs')
    
    dataset['Title'] = dataset['Title'].map(title)
    dataset['Title'] = dataset['Title'].fillna(0)
    
df = pd.concat(data, axis=0, ignore_index=True)


df = df.drop(['Name'], axis=1)


# In[159]:


df.info()


# In[160]:


df.head()


# In[161]:


genders = {"male": 1, "female": 2}
df['Sex'] = df['Sex'].map(genders)


# In[162]:


df['Ticket'].describe()


# In[163]:


df = df.drop(['Ticket'],axis = 1)


# In[164]:


df.head(10)


# In[165]:


ports = {'S': 0, 'C': 1, 'Q': 2}
df['Embarked'] = df['Embarked'].map(ports)


# In[166]:


df.loc[df['Age'] <= 11, 'Age'] = 0
df.loc[(df['Age'] > 11) & (df['Age'] <= 18), 'Age'] = 1
df.loc[(df['Age'] > 18) & (df['Age'] <= 22), 'Age'] = 2
df.loc[(df['Age'] > 22) & (df['Age'] <= 27), 'Age'] = 3
df.loc[(df['Age'] > 27) & (df['Age'] <= 33), 'Age'] = 4
df.loc[(df['Age'] > 33) & (df['Age'] <= 40), 'Age'] = 5
df.loc[(df['Age'] > 40) & (df['Age'] <= 66), 'Age'] = 6
df.loc[df['Age'] > 66, 'Age'] = 6

df['Age'].value_counts()


# In[167]:


df.head(10)


# In[196]:


data = pd.DataFrame(data, columns=['Fare'])
data.loc[data['Fare'] <= 7.91, 'Fare'] = 0
data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare'] = 2
data.loc[(data['Fare'] > 31) & (data['Fare'] <= 99), 'Fare'] = 3
data.loc[(data['Fare'] > 99) & (data['Fare'] <= 250), 'Fare'] = 4
data.loc[data['Fare'] > 250, 'Fare'] = 5


# In[197]:


data = df
data['Age_Class']  = data['Age']* data['Pclass']


# In[198]:


data['Fare_per_Person'] = data['Fare']/ (data['relatives']+1)
data['Fare_per_Person'] = data['Fare_per_Person'].astype(int)
data.head(10)


# In[199]:


x =df.drop('Survived', axis=1)
y  = df['Survived']


# In[200]:


print(df.columns)


# In[201]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)


# In[202]:


x


# In[203]:


y


# In[204]:


log = LogisticRegression()


# In[205]:


log.fit(x_train,y_train)


# In[206]:


y_pred = log.predict(x_test)


# In[207]:


y_pred


# In[208]:


acc = round(log.score(x_train,y_train)*100,2)


# In[209]:


dec = DecisionTreeClassifier()
dec.fit(x_train,y_train)
y_pred = dec.predict(x_test)
accuracy = round(dec.score(x_train,y_train) * 100 ,2)


# In[210]:


random = RandomForestClassifier(n_estimators=100)
random.fit(x_train,y_train)


y_pred_random = random.predict(x_test)

random.score(x_train, y_train)
acc_random = round(random.score(x_train , y_train) * 100 , 2)


# In[211]:


linear = LinearSVC()
linear.fit(x_train,y_train)
y_predict = linear.predict(x_test)
acc_linear_svc = round(linear.score(x_train,y_train) * 100 , 2)


# In[212]:


result = pd.DataFrame({
    'Model': ["Support Vector Machine" , "Logistic Regression",
             "Random Forest", "Decision Tree"],
    'Score': [acc_linear_svc,acc,
             acc_random ,accuracy]})
result_df = result.sort_values(by = 'Score' , ascending = False)
result_df = result_df.set_index("Score")
result_df.head(9)


# In[213]:


from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier(n_estimators=100)
score = cross_val_score(rf , x_train , y_train , cv = 10 , scoring="accuracy")
print("Scores",score)
print("Mean:",score.mean())
print("Standard Devation:", score.std())


# In[214]:


importances = pd.DataFrame({'feature': x_train.columns,'importance':np.round(random.feature_importances_,3)})
importances = importances.sort_values('importance', ascending=False).set_index("feature")
importances.head(10)


# In[215]:


importances.plot.bar()


# In[216]:


random = RandomForestClassifier(criterion='gini',
                                min_samples_leaf=1,
                                min_samples_split=10,
                                n_estimators = 100,
                                max_features='sqrt',
                                oob_score=True,
                                random_state=1,
                                n_jobs=1)
random.fit(x_train,y_train)
y_predictions = random.predict(x_test)
random.score(x_train,y_train)

print('oob score', round(random.oob_score_,4) * 100 , '%')


# In[217]:


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
predictions= cross_val_predict(log , x_train , y_train , cv = 3)
confusion_matrix(y_train , predictions)


# In[218]:


from sklearn.metrics import precision_score , recall_score
print("Precision score:" , precision_score(y_train , predictions))
print("Recall Score :" , recall_score(y_train , predictions))


# In[219]:


from sklearn.metrics import f1_score
print("F1 Score :" , f1_score(y_train , predictions))


# In[220]:


y_scores = random.predict_proba(x_train)
y_scores = y_scores[:, 1]

precision, recall, threshold = precision_recall_curve(y_train, y_scores)

def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r-", label='precision', linewidth=5)
    plt.plot(threshold, recall[:-1], 'b', label='recall', linewidth=5)
    plt.xlabel("Threshold", fontsize=20)
    plt.legend(loc="upper right", fontsize=20)
    plt.ylim([0, 1])

plt.figure(figsize=(14, 7))
plot_precision_and_recall(precision, recall, threshold)
plt.show()


# In[221]:


from sklearn.metrics import roc_curve
false_positive_rate , true_positive , thresholds = roc_curve(y_train , y_scores)
def plot_roc_curve(false_positive_rate,true_positive, label = None):
    plt.plot(false_positive_rate , true_positive ,linewidth = 2 , label = label)
    plt.plot([0, 1], [0, 1], 'r' , linewidth = 4)
    plt.axis([0 ,1,0, 1])
    plt.xlabel("False Positive Rate (FPR)" , fontsize = 18)
    plt.ylabel("True Positive Rate (TPR)" , fontsize = 18)

plt.figure(figsize= (14 ,7))
plot_roc_curve(false_positive_rate , true_positive )
plt.show()


# In[223]:


from sklearn.metrics import roc_auc_score
rather = roc_auc_score(y_train , y_scores)
print("ROC SCORES :", rather)


# In[ ]:





# In[ ]:




