
# coding: utf-8

# In[ ]:


import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from keras.models import Sequential
from keras.layers import Dense , Dropout , Lambda, Flatten
from keras.optimizers import Adam ,RMSprop
from sklearn.model_selection import train_test_split
from keras import  backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras import to_categorical


# In[ ]:


os.chdir('E:\Data\digit-recognizer')
train = pd.read_csv('train.csv')
test=pd.read.csv('test.csv')


# In[ ]:


Y_train = train["label"]

# Drop 'label' column
X_train = train.drop(labels = ["label"],axis = 1) 

g = sns.countplot(Y_train)

Y_train.value_counts()


# In[ ]:


# Normalize the data
X_train = X_train / 255.0
test = test / 255.0


# In[ ]:


#one more dimesnsion for keras as it is a grayscale
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)


# In[ ]:


# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
Y_train = to_categorical(Y_train, num_classes = 10)


# In[ ]:


#define the model
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5), 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5), 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


# In[ ]:


#converting into 1D array
model.add(Flatten())

#final fully connected layer
model.add(Dense(10, activation = "softmax"))

#further will be using optimizer functions and train the model with epocs and check the error rate

