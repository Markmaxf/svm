
# coding: utf-8

# In[137]:


import pandas as pd
import matplotlib.pyplot as plt 
get_ipython().magic('matplotlib inline')
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np 
from sklearn.preprocessing import StandardScaler


# In[248]:


# generate example for svm analysis

def linear_data(sample_size,coff):
    weight=coff[0]
    bias=coff[1]
    
    noise=np.random.rand(sample_size)
    x1=np.random.rand(sample_size)*100
    x2=np.random.rand(sample_size)*100
    y=x1*weight+ bias + noise

    
    label=[1  if y[i]>x2[i] else 0 for i in range(sample_size) ]
    return np.transpose(np.vstack((x1,x2))),label

def exp_data(sample_size,coff):
    coff_1=coff[0]
    coff_2=coff[1]
    
    half_size=int(sample_size/2)
    x1_pos=np.random.rand(half_size)*10
    x1_neg=np.random.rand(sample_size-half_size)*10
    
    x2_pos=np.random.rand(half_size)*10
    x2_neg=np.random.rand(sample_size-half_size)*10
    
    x1=np.hstack((x1_pos,x1_neg))
    x2=np.hstack((x2_pos,x2_neg))
    
    noise=np.random.rand(sample_size)
    
    label=[1 if x1[i]**2+x2[i]**2+noise[i]>48 else 0 for i in range(sample_size)  ]
    
    return np.transpose(np.vstack((x1,x2))),label
    
    


# In[269]:


x,y=linear_data(1000,[2,0])


# In[272]:


svc=SVC(C=10,kernel='linear')

sc=StandardScaler()
#x=sc.fit_transform(x)

svc.fit(x,y)


# In[295]:


vector=svc.support_vectors_

x1_min=min(x[:,0])
x1_max=max(x[:,0])

x2_min=min(x[:,1])
x2_max=max(x[:,1])

x1,x2=np.meshgrid(np.linspace(x1_min,x1_max,num=50),np.linspace(x1_min,x2_max,num=50))
plt.contourf(x1,x2,svc.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
            alpha=0.5,cmap=ListedColormap(('red','blue')))
plt.xlim(x1.min(),x2.max())
plt.ylim(x2.min(),x2.max())



plt.scatter(x[:,0],x[:,1],c=y)
plt.scatter(vector[:,0],vector[:,1],c='red',s=130,marker='s')
plt.ylim([0,max(x[:,1])])


# In[287]:


svc.support_vectors_


# In[307]:


x,y=exp_data(1000,[2,0])

svc=SVC(C=0.5,kernel='rbf')

#sc=StandardScaler()
#x=sc.fit_transform(x)

svc.fit(x,y)


# In[308]:


vector=svc.support_vectors_


x1_min=min(x[:,0])
x1_max=max(x[:,0])

x2_min=min(x[:,1])
x2_max=max(x[:,1])

x1,x2=np.meshgrid(np.linspace(x1_min,x1_max,num=50),np.linspace(x1_min,x2_max,num=50))
plt.contourf(x1,x2,svc.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
            alpha=0.5,cmap=ListedColormap(('red','blue')))
plt.xlim(x1.min(),x2.max())
plt.ylim(x2.min(),x2.max())

plt.scatter(x[:,0],x[:,1],c=y)
#plt.scatter(vector[:,0],vector[:,1],c='red',s=30,marker='s')
plt.ylim([0,max(x[:,1])])


# In[309]:


vector[:,0]


# In[ ]:


x

