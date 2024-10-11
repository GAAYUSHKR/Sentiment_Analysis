
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import wordcloud as WordCloud
import nltk
import re
import string


# In[2]:


train=pd.read_csv("file:///C:/Users/Avinash/Downloads/all/train.tsv",sep='\t')
test=pd.read_csv("file:///C:/Users/Avinash/Downloads/all/test.tsv",sep='\t')


# In[3]:


train.head()


# In[4]:


print(test.shape)
train.shape


# In[5]:


combi=train.append(test,ignore_index=True)


# In[6]:


combi.shape


# In[8]:


from nltk.corpus import stopwords


# In[9]:


stopword=set(stopwords.words("english"))


# In[10]:


combi['clean_review']=combi['Phr;ase'].apply(lambda x:' '.join([w for w in x.split() if w not in stopword]))


# In[11]:


combi.head()


# In[13]:


from nltk.tokenize import word_tokenize


# In[18]:


tokenized_review=combi['Phrase'].apply(lambda x: x.split())


# In[19]:


tokenized_review


# In[12]:


#combi['clean_review']=combi.apply(lambda x: word_tokenize(x['clean_review']),axis=1)


# In[13]:


#combi.head()


# In[17]:


from nltk.stem import WordNetLemmatizer
lemma=WordNetLemmatizer()
from nltk.stem import PorterStemmer
stemmer=PorterStemmer()


# In[58]:


#combi['clean_review']=combi['clean_review'].apply(lambda x: [lemma.lemmatize(i) for i in x])


# In[20]:


tokenized_review=tokenized_review.apply(lambda x: [stemmer.stem(i) for i in x])


# In[21]:


for i in range(len(tokenized_review)):
    tokenized_review[i]=' '.join(tokenized_review[i])


# In[22]:


tokenized_review


# In[23]:


combi['clean_review']=tokenized_review


# In[24]:


combi.head()


# In[30]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


# In[34]:


bow_vect=CountVectorizer(max_df=0.90,min_df=2,max_features=1000,stop_words='english')
bow=bow_vect.fit_transform(combi['clean_review'])


# In[40]:


train_bow=bow[:156060,:]
test_bow=bow[156060:,:]


# In[43]:


x_train,y_train,x_label,y_label=train_test_split(train_bow,train['Sentiment'],test_size=0.2)


# In[50]:


lreg=LogisticRegression()
lreg.fit(x_train,x_label)


# In[52]:


prediction=lreg.predict(y_train)


# In[55]:


from sklearn.metrics import accuracy_score
accuracy_score(y_label,prediction)


# In[57]:


test_pred=lreg.predict(test_bow)


# In[59]:


submission=pd.read_csv("file:///C:/Users/Avinash/Downloads/all/sampleSubmission.csv")


# In[61]:


submission.head()


# In[63]:


submission['Sentiment']=test_pred


# In[65]:


submission.head()


# In[67]:


pd.DataFrame(submission, columns=['Phraseid','Sentiment']).to_csv('movie.csv')

