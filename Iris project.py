#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


pip install ucimlrepo


# In[4]:


from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
X = iris.data.features 
y = iris.data.targets 
  
# metadata 
print(iris.metadata) 
  
# variable information 
print(iris.variables) 


# In[5]:


df = pd.read_csv('/Users/aekankshtarware/Downloads/Iris Species/Iris.csv')


# In[6]:


df.head()


# In[7]:


# delete a column
df = df.drop(columns = ['Id'])
df.head()


# In[8]:


# to display stats of data
df.describe()


# In[9]:


df.info()


# In[10]:


# to display no. of samples in each class

df['Species'].value_counts()


# # Pre processing

# In[11]:


# check for null values

df.isnull().sum()


# # Exploratory Data Analysis

# In[12]:


df['SepalLengthCm'].hist()


# In[13]:


df['SepalWidthCm'].hist()


# In[14]:


df['PetalLengthCm'].hist()


# In[15]:


df['PetalWidthCm'].hist()


# In[16]:


# scatterplot

colors = ['red', 'orange', 'blue']
species = ['Iris-virginica', 'Iris-versicolor', 'Iris-setosa']


# In[17]:


for i in range(3):
    x = df[df['Species']== species[i]]
    plt.scatter(x['SepalLengthCm'], x['SepalWidthCm'], c = colors[i], label = species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()


# In[18]:


for i in range(3):
    x = df[df['Species']== species[i]]
    plt.scatter(x['PetalLengthCm'], x['PetalWidthCm'], c = colors[i], label = species[i])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()


# In[19]:


for i in range(3):
    x = df[df['Species']== species[i]]
    plt.scatter(x['SepalLengthCm'], x['PetalLengthCm'], c = colors[i], label = species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend()


# In[20]:


for i in range(3):
    x = df[df['Species']== species[i]]
    plt.scatter(x['PetalLengthCm'], x['SepalLengthCm'], c = colors[i], label = species[i])
plt.xlabel("Petal Length")
plt.ylabel("Sepal Length")
plt.legend()


# In[21]:


for i in range(3):
    x = df[df['Species']== species[i]]
    plt.scatter(x['SepalWidthCm'], x['PetalWidthCm'], c = colors[i], label = species[i])
plt.xlabel("Sepal Width")
plt.ylabel("Petal Width")
plt.legend()


# In[22]:


for i in range(3):
    x = df[df['Species']== species[i]]
    plt.scatter(x['PetalWidthCm'], x['SepalWidthCm'], c = colors[i], label = species[i])
plt.xlabel("Petal Width")
plt.ylabel("Sepal Width")
plt.legend()


# # Cooreation Matrix

# In[24]:


df.corr()


# In[27]:


corr = df.corr()
fig, ax = plt.subplots(figsize=(5,5))
sns.heatmap(corr, annot = True, ax = ax, cmap = 'coolwarm')


# # Label Encoder

# In[28]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[29]:


df["Species"] = le.fit_transform(df['Species'])
df.head()


# # Model Training

# In[66]:


from sklearn.model_selection import train_test_split
# Train- 70%
# Test-  30%
X = df.drop(columns=['Species'])
Y = df['Species']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30)


# In[56]:


# logistic regression REMEBER TO MAKE IT CALLABLE
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[57]:


# model train
model.fit(X_train, Y_train)


# In[67]:



# print metric to get performace

print("Accuracy:", model.score(X_test, Y_test)*100)


# In[59]:


# KNN K Nearest Neighbour
from sklearn.neighbors import KNeighborsClassifier


# In[60]:


model = KNeighborsClassifier()


# In[61]:


model.fit(X_train, Y_train)


# In[68]:


# print metric to get performace

print("Accuracy:", model.score(X_test, Y_test)*100)


# In[63]:


# Decision Tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


# In[64]:


model.fit(X_train, Y_train)


# In[69]:


# print metric to get performace

print("Accuracy:", model.score(X_test, Y_test)*100)


# In[ ]:




