#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


data = {'time':['6','7','8','9','10','11','12','13'],'temperature(c)':[22,24,26,27,28,29,30,31]}
data


# In[5]:


climate = pd.DataFrame(data,columns=['time','temperature(c)'])
climate


# In[6]:


climate.to_csv("C:/Users/hp/Downloads/climate.csv")


# In[7]:


pd.read_csv("C:/Users/hp/Downloads/climate.csv")


# In[8]:


climate = pd.DataFrame(data,columns=['time','temperature(c)'])
climate


# In[9]:


climate.plot.scatter(x='time',y='temperature(c)')


# In[10]:


x=climate[['time']]
x


# In[11]:


y=climate[['temperature(c)']]
y


# In[12]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.5,random_state=0)
xtrain


# In[13]:


ytrain


# In[14]:


from sklearn.linear_model import LinearRegression


# In[15]:


lr=LinearRegression()
lr


# In[16]:


lr.fit(xtrain,ytrain)


# In[17]:


prediction = lr.predict(xtest)
prediction


# In[18]:


prediction=pd.DataFrame(prediction,columns=['prediction'])
prediction


# In[ ]:




