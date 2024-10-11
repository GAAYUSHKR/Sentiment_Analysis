
# coding: utf-8

# In[1]:


import re
import pandas as pd
import nltk
import string
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 


# In[2]:


train=pd.read_csv("file:///C:/Users/Avinash/Downloads/tweets_train_E6oV3lV.csv")
test=pd.read_csv("file:///C:/Users/Avinash/Downloads/test_tweets_anuFYb8.csv")


# In[3]:


print(test.shape)
train.shape


# In[4]:


train.head()


# In[5]:


test.head()


# In[6]:


combi=train.append(test,ignore_index=True)


# In[7]:


combi.head()


# In[8]:


def remove_pattern(input_txt,pattern):
    r=re.findall(pattern,input_txt)
    for i in r:
        input_txt=re.sub(i,'',input_txt)
        
    return input_txt


# In[9]:


combi['tidy_tweet']=np.vectorize(remove_pattern)(combi['tweet'],"@[\w]*")


# In[10]:


combi.head()


# In[11]:


combi['tidy_tweet']=combi['tidy_tweet'].str.replace("[^a-zA-Z#]","  ")


# combi.head()

# In[12]:


combi.head()


# In[13]:


combi['tidy_tweet']=combi['tidy_tweet'].apply(lambda x:'  '.join([w for w in x.split() if len(w)>3]))


# In[14]:


combi.head()


# In[15]:


from nltk.tokenize import word_tokenize


# In[16]:


tokenized_tweet=combi['tidy_tweet'].apply(lambda x: x.split())


# In[18]:


combi['tidy_tweet_1'] = combi.apply(lambda row: word_tokenize(row['tidy_tweet']),axis=1)


# In[19]:


combi.head()


# In[20]:


from nltk.stem.porter import*
stemmer=PorterStemmer()


# In[21]:


tokenized_tweet=tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])


# In[23]:


combi['tidy_tweet_1']=combi['tidy_tweet_1'].apply(lambda x: [stemmer.stem(i)for i in x])


# In[24]:


combi.head()


# In[25]:



for i in range(len(tokenized_tweet)):
    tokenized_tweet[i]=tokenized_tweet[i]
    


# In[26]:



for i in range(len(tokenized_tweet)):
    tokenized_tweet[i]=' '.join(tokenized_tweet[i])


# In[28]:


combi['tidy_tweet']=tokenized_tweet


# In[29]:


combi.head()


# In[30]:


combi['tidy_tweet'].dtypes


# In[31]:


all_words = ' '.join(' '.join(txt) for txt in combi['tidy_tweet'])


# In[37]:


racist_words = ' '.join(''.join(txt) for txt in combi['tidy_tweet'][combi['label']==1])


# In[1]:


from wordcloud import WordCloud


# In[34]:


wordcloud=WordCloud(width=800,height=500,random_state=21,max_font_size=110).generate(racist_words)


# In[35]:


plt.imshow(wordcloud,interpolation="bilinear")
plt.figure(figsize=(20,20))
plt.show()


# In[40]:


from sklearn.feature_extraction.text import CountVectorizer


# In[41]:


bow_vect = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = bow_vect.fit_transform(combi['tidy_tweet'])


# In[42]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


# In[47]:


train_bow=bow[:31962,:]
test_bow=bow[31962:,:]


# In[75]:


xtrain,xvalid,ytrain,yvalid=train_test_split(train_bow,train['label'],test_size=0.3)


# In[82]:


yvalid.dtype


# In[54]:


lreg=LogisticRegression()
lreg.fit(xtrain,ytrain)


# In[55]:


prediction=lreg.predict(xvalid)


# In[71]:


prediction.dtype


# In[58]:


f1_score(yvalid, prediction)


# In[88]:


test_pred=lreg.predict(test_bow)


# In[89]:


test_pred


# In[90]:


test['label']=test_pred


# In[92]:


test

