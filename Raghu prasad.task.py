#!/usr/bin/env python
# coding: utf-8

# In[190]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[191]:


import os
print(os.getcwd())


# In[193]:


quality = pd.read_csv('AirQualityUCI.csv')
quality


# # data cleaning

# In[194]:


quality1=quality.replace(-200,np.nan)
quality1


# In[195]:


quality1.dropna()


# In[196]:


quality1.head()


# In[197]:


quality1.shape


# # visualisation: relation between every other attribute 

# In[198]:


import seaborn as sns
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
sns.set(style='whitegrid', context='notebook')
features_plot = ['NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)',
            'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)',
             'T','RH', 'AH']

data_to_plot = air_data1[features_plot]
data_to_plot = scalar.fit_transform(data_to_plot)
data_to_plot = pd.DataFrame(data_to_plot)

sns.pairplot(data_to_plot, size=2.0);
plt.tight_layout()
plt.show()


# In[199]:


import seaborn as sns
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
sns.set(style='whitegrid', context='notebook')
features_plot = ['C6H6(GT)', 'RH', 'AH', 'PT08.S1(CO)']

data_to_plot = quality1[features_plot]
data_to_plot = scalar.fit_transform(data_to_plot)
data_to_plot = pd.DataFrame(data_to_plot)

sns.pairplot(data_to_plot, size=2.0);
plt.tight_layout()
plt.show()


# In[ ]:





# In[200]:


features = quality1


# In[201]:


features = features.drop('Date', axis=1)
features = features.drop('Time', axis=1)
features = features.drop('C6H6(GT)', axis=1)
features = features.drop('PT08.S4(NO2)', axis=1)
features = features.drop('RH', axis=1)


# In[202]:


labels = quality1['RH'].values


# In[203]:


features = features.values


# # cross validation: splitting the data

# In[204]:


from sklearn.model_selection import train_test_split
#pip install -U scikit-learn
X_train,X_test,y_train,y_test = train_test_split(features,labels,test_size=0.3)
print('Partitioning Done!')


# In[205]:


print("X_trian shape --> {}".format(X_train.shape))
print("y_train shape --> {}".format(y_train.shape))
print("X_test shape --> {}".format(X_test.shape))
print("y_test shape --> {}".format(y_test.shape))


# In[206]:





# In[ ]:




