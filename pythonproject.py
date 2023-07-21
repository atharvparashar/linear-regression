#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('E:\CONTANT/advertising.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[5]:


df = pd.read_csv("E:\CONTANT/advertising.csv")
df.head()


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


attributes = ['TV', 'Radio', 'Newspaper']

fig, axes = plt.subplots(3, 1)

for i, attr in enumerate(attributes):
    sns.scatterplot(data=df, x=attr, y='Sales', ax=axes[i])

    axes[i].set_title(f'Sales vs {attr}')
plt.show()
#Correlation Matrix
sns.heatmap(df.corr(), annot = True)
plt.show()


# In[9]:


null_values = list(df.isnull().sum())
columns= list(df.columns)
columns
count = 0
for column in columns:
    print(column + " has " + str(null_values[count]) + " null values.")
    count += 1


# In[10]:


fig, axs = plt.subplots(3, figsize = (5,5))
plt1 = sns.boxplot(df['TV'], ax = axs[0])
plt2 = sns.boxplot(df['Newspaper'], ax = axs[1])
plt3 = sns.boxplot(df['Radio'], ax = axs[2])


# In[12]:


def call_out(df):
    lst=df.select_dtypes(exclude='object').columns.tolist() #Only remove outliers for numerical columns
    for i in lst:
        if i != 'sales': #ignore the target attribute
            df=outlier(df,df[i])
    return df

#Function to remove attribute
def outlier(df,data_column):
    from scipy.stats import skew
    import statistics

    qi=data_column.quantile(0.25) #Calculate the first 25% of the dataset
    qf=data_column.quantile(0.75) #Calculate the last 25% of the dataset
    iqr=qf-qi #Calculate Inter quantile range
    c=1.5
    #Calculate upper and lower limit for outlier detection
    upper_limit=qf+c*iqr
    lower_limit=qi-c*iqr
    
    arr=data_column.to_numpy()
    med=statistics.median(arr) #Calculate the median for that attributee

    for i in data_column:            
        if((i<lower_limit) | (i>upper_limit)):
            data_column=data_column.replace(i,med) #if value is outside of the given range then replace it with the median for that attribute
    return df

df = call_out(df = df)


# In[13]:


X = df[['TV']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
print("Training MAE:", train_mae.round(3))
print("Testing MAE:", test_mae.round(3))
print("Training R2 score:", train_r2.round(3))
print("Testing R2 score:", test_r2.round(3))

plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='green', label='Testing Data')
plt.plot(X_train, y_train_pred, color='red', linewidth=2, label='Training Regression Line')
plt.plot(X_test, y_test_pred, color='orange', linewidth=2, label='Testing Regression Line')
plt.xlabel('TV')
plt.ylabel('Sales')
plt.legend()
plt.show()


# In[14]:


X = df[['Radio']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("Training MAE:", train_mae.round(3))
print("Testing MAE:", test_mae.round(3))
print("Training R2 score:", train_r2.round(3))
print("Testing R2 score:", test_r2.round(3))

plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='green', label='Testing Data')
plt.plot(X_train, y_train_pred, color='red', linewidth=2, label='Training Regression Line')
plt.plot(X_test, y_test_pred, color='orange', linewidth=2, label='Testing Regression Line')
plt.xlabel('TV')
plt.ylabel('Sales')
plt.legend()
plt.show()


# In[15]:


X = df[['Newspaper']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("Training MAE:", train_mae.round(3))
print("Testing MAE:", test_mae.round(3))
print("Training R2 score:", train_r2.round(3))

print("Testing R2 score:", test_r2.round(3))

plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='green', label='Testing Data')
plt.plot(X_train, y_train_pred, color='red', linewidth=2, label='Training Regression Line')
plt.plot(X_test, y_test_pred, color='orange', linewidth=2, label='Testing Regression Line')
plt.xlabel('TV')
plt.ylabel('Sales')
plt.legend()
plt.show()


# In[16]:


X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("Training MAE:", train_mae.round(3))
print("Testing MAE:", test_mae.round(3))

print("Training R2 score:", train_r2.round(3))
print("Testing R2 score:", test_r2.round(3))


# In[ ]:




