#!/usr/bin/env python
# coding: utf-8

# ### Importing the dependencies

# In[8]:


import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


# ### Data Collection and Pre-Processing

# In[9]:


# loading the data from csv file to a pandas dataframe

raw_mail_data = pd.read_csv('mail_data.csv')


# In[10]:


print(raw_mail_data)


# In[11]:


# replace the null values with a null string

mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')


# In[12]:


# printing the first 5 rows of the dataframe

mail_data.head()


# In[13]:


# checking the number of rows and columns in the dataframe

mail_data.shape


# ### Label Encoding

# In[14]:


# label spam mail as 1; ham (non-spam) mail as 0

mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 1
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 0


# ### Spam = 1; Ham = 0

# In[15]:


# separating the data as texts and labels

X = mail_data['Message']
Y = mail_data['Category']


# In[16]:


print(X)


# In[17]:


print(Y)


# ### Splitting the data into training data and test data

# In[18]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 3)


# In[19]:


print(X.shape)
print(X_train.shape)
print(X_test.shape)


# ### Feature Extraction

# In[20]:


# transform the text data to feature vectors that can be used as input to the logistic regression

feature_extraction = TfidfVectorizer(min_df = 1, stop_words = 'english', lowercase = 'True')

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# convert Y_train and Y_test values as integers

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# In[21]:


X_train_features


# In[22]:


print(X_train_features)


# ### Training the Logistic Regression Model

# In[23]:


model = LogisticRegression()


# In[24]:


# training the model with training data

model.fit(X_train_features, Y_train)


# ### Evaluating the trained model

# In[25]:


# prediction on training data

prediction_train = model.predict(X_train_features)
accuracy_train = accuracy_score(Y_train, prediction_train)


# In[26]:


print('Accuracy on training data: ', accuracy_train)


# In[27]:


# prediction on test data

prediction_test = model.predict(X_test_features)
accuracy_test = accuracy_score(Y_test, prediction_test)


# In[28]:


print('Accuracy on test data: ', accuracy_test)


# In[29]:


# f1 scores

print(f1_score(Y_test, prediction_test))
print(f1_score(Y_test, prediction_test, average=None, labels=[1, 0]))


# ### Building a Predictive System

# In[30]:


input_mail = ["500 New Mobiles from 2004, MUST GO! Txt: NOKIA to No: 89545 & collect yours today!From ONLY £1 www.4-tc.biz 2optout 087187262701.50gbp/mtmsg18", 
              'Tried calling you several times but your phone was busy. Please call me back asap']


# In[31]:


Y_test.iloc[13]


# In[32]:


X_test.iloc[13]


# In[33]:


# converting text to feature vectors

input_mail_features = feature_extraction.transform([X_test.iloc[13], input_mail[0], input_mail[1]])

# making prediction

prediction = model.predict(input_mail_features)
pred = []

for val in prediction:
    if val == 1:
        pred.append('Spam')
    else:
        pred.append('Ham')

print(pred)


# ### Saving the model

# Save Model

# In[34]:


with open('spam_classifier.pkl', 'wb') as f:
    pickle.dump(model, f)


# Load Model

# In[35]:


with open('spam_classifier.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

