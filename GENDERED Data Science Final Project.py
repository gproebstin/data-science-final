#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import packages needed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from nltk.tokenize import RegexpTokenizer

#from scipy.
import pandas as pd
import numpy as np

import random


# ## Import word embeddings

# In[2]:


import gensim.downloader
glove_twitter_50 = gensim.downloader.load('glove-twitter-100')


# ## Read in data

# In[3]:


get_ipython().system('ls ../data')


# In[4]:


#df = pd.read_csv("../data/train_all_tasks.csv")
df = pd.read_csv("../data/train_all_tasks.csv")
df_girl_names = pd.read_csv("../data/girl_names_2000.csv")
df_boy_names = pd.read_csv("../data/boy_names_2000.csv")

# top_ten_boy_names = df_boy_names[:10]
# top_ten_girl_names = df_girl_names[:10]

top_ten_girl_names = pd.DataFrame(data={'Name':["Emily", "Madison", "Emma", "Olivia", "Hannah", "Abigail", "Isabella", "Samantha", "Elizabeth", "Ashley"]})
top_ten_boy_names = pd.DataFrame(data={'Name':["Jacob", "Michael", "Joshua", "Matthew", "Daniel", "Christopher", "Andrew", "Ethan", "Joseph", "William"]})


small_df = df[['text', 'label_sexist']] # only use necssary columns

top_ten_girl_names['Name']
top_ten_boy_names['Name']


# ### Data statistics

# In[5]:


from sklearn.model_selection import StratifiedShuffleSplit

X = small_df['text']
#print(small_df['label_sexist'].value_counts(normalize=True))
y = small_df['label_sexist']

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=1)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

train_df = pd.concat([X_train, y_train], axis=1, keys=['text', 'label_sexist'])
test_df = pd.concat([X_test, y_test], axis=1, keys=['text', 'label_sexist'])

print("train shape is", train_df.shape)
print("test shape is", test_df.shape)


test_df['label_sexist'].value_counts(normalize=True)


# ## make new gendered data set

# ### female names:

# In[6]:


# new pd data frame w/ four columns --> tweet, class, name, and gender?
#Should I remove the "!!! RT @someusername"
female_labeled_array =[]
male_labeled_array =[]

for row in train_df.iterrows():
    tweet_string = row[1]['text']
    class_label = row[1]['label_sexist']
    if "not" in class_label:
        class_label = 0
    else:
        class_label = 1
    for f_name_string in top_ten_girl_names['Name']:
        new_tweet_string = f_name_string + " posted " + tweet_string
        female_sentence_arr = [new_tweet_string, class_label, f_name_string, "f"] #using an array instead of a dictionary because dictionaries are slow for some reason
        female_labeled_array.append(female_sentence_arr)
    for m_name_string in top_ten_boy_names['Name']:
        new_tweet_string = m_name_string + " posted " + tweet_string
        male_sentence_arr = [new_tweet_string, class_label, m_name_string, "m"] #using an array instead of a dictionary because dictionaries are slow for some reason
        male_labeled_array.append(male_sentence_arr)
        
        
female_test_arr = []
male_test_arr = []

for row in test_df.iterrows():
    tweet_string = row[1]['text']
    class_label = row[1]['label_sexist']
    if "not" in class_label:
        class_label = 0
    else:
        class_label = 1
    for f_name_string in top_ten_girl_names['Name']:
        new_tweet_string = f_name_string + " posted " + tweet_string
        female_sentence_arr = [new_tweet_string, class_label, f_name_string, "f"] #using an array instead of a dictionary because dictionaries are slow for some reason
        female_test_arr.append(female_sentence_arr)
    for m_name_string in top_ten_boy_names['Name']:
        new_tweet_string = m_name_string + " posted " + tweet_string
        male_sentence_arr = [new_tweet_string, class_label, m_name_string, "m"] #using an array instead of a dictionary because dictionaries are slow for some reason
        male_test_arr.append(male_sentence_arr)

np_male = np.array(male_labeled_array)
print(np_male[0])
print(np_male.shape)

np_female = np.array(female_labeled_array)
print(np_female[0])
print(np_female.shape)

np_test_male = np.array(male_test_arr)
print(np_test_male[0])
print(np_test_male.shape)

np_test_female = np.array(female_test_arr)
print(np_test_female[0])
print(np_test_female.shape)


# ## make FEMALE  X and y TRAIN sets

# In[7]:


#okay actually before I worry about this why don't I just run two different vectorizers/trains on my two arrays...

X_female = []
y_female_train = []

for sentence_arr in female_labeled_array:
    X_female.append(sentence_arr[0])
    y_female_train.append(sentence_arr[1])

tokenizer = RegexpTokenizer(r'\w+')

#df_X_female = pd.DataFrame(X_female)

tokenized_arr = []
for tweet_sentence in X_female:
    tokenized_sentence = tokenizer.tokenize(tweet_sentence.lower())
    tokenized_arr.append(tokenized_sentence)

print(tokenized_arr[0])

list_of_word_arrs = []

for word_arr in tokenized_arr:
    existing_keys_arr = []
    #print(word_arr)
    for word in word_arr:
        if word in glove_twitter_50.key_to_index:
            existing_keys_arr.append(word)
    list_of_word_arrs.append(existing_keys_arr)
    
embed_arr = []
i=0
empty_indices = []
for word_arr in list_of_word_arrs:
    #print(word_arr)
    try:
        embed_arr.append(glove_twitter_50[word_arr])
    except: #TODO: CHANGE THIS
        empty_indices.append(i)
        pass
    i = i + 1
print(empty_indices)

# new_y_female = []

# j = 0
# for element in y_female:
#     if not j in empty_indices:
#         new_y_female.append(element)
#     j = j + 1


female_train_embed_list = []
for sentence_word_embed_list in embed_arr:
    np_sentence_list = np.array(sentence_word_embed_list)
    np_sum_arr = np.sum(np_sentence_list, axis=0)
    #print(np_sum_arr.shape[0])
    sum_arr = np_sum_arr.tolist()
    female_train_embed_list.append(sum_arr)
    
#np_word_embedding = np.array(embed_arr[0])
np_female_train_embed_list = np.array(female_train_embed_list)
np_female_train_embed_list.shape


# ## Make FEMALE X and y TEST sets

# In[8]:


X_female = []
y_female_test = []

for sentence_arr in female_test_arr: #this was the change between train and test
    X_female.append(sentence_arr[0])
    y_female_test.append(sentence_arr[1])

tokenizer = RegexpTokenizer(r'\w+')

#df_X_female = pd.DataFrame(X_female)

tokenized_arr = []
for tweet_sentence in X_female:
    tokenized_sentence = tokenizer.tokenize(tweet_sentence.lower())
    tokenized_arr.append(tokenized_sentence)

print(tokenized_arr[0])

list_of_word_arrs = []

for word_arr in tokenized_arr:
    existing_keys_arr = []
    #print(word_arr)
    for word in word_arr:
        if word in glove_twitter_50.key_to_index:
            existing_keys_arr.append(word)
    list_of_word_arrs.append(existing_keys_arr)
    
embed_arr = []
i=0
empty_indices = []
for word_arr in list_of_word_arrs:
    #print(word_arr)
    try:
        embed_arr.append(glove_twitter_50[word_arr])
    except: #TODO: CHANGE THIS
        empty_indices.append(i)
        pass
    i = i + 1
print(empty_indices)

# new_y_female = []

# j = 0
# for element in y_female:
#     if not j in empty_indices:
#         new_y_female.append(element)
#     j = j + 1


female_test_embed_list = []
for sentence_word_embed_list in embed_arr:
    np_sentence_list = np.array(sentence_word_embed_list)
    np_sum_arr = np.sum(np_sentence_list, axis=0)
    sum_arr = np_sum_arr.tolist()
    female_test_embed_list.append(sum_arr)
    
#np_word_embedding = np.array(embed_arr[0])
np_female_test_embed_list = np.array(female_test_embed_list)
np_female_test_embed_list.shape


# In[9]:


# okay so there are three missing values in the embed_train-- these are presumably the values I deleted?
clf = LogisticRegression(max_iter = 2000)
clf.fit(female_train_embed_list, y_female_train)
print("model accuracy on the TRAIN:", sum(clf.predict(female_train_embed_list) == y_female_train) / len(y_female_train))


# ## confusion matrix

# In[10]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# female_test_predictions = clf.predict(female_test_embed_list)
# cm_female = confusion_matrix(y_female_test, female_test_predictions)
# print("TEST accuracy:", sum(clf.predict(female_test_embed_list) == y_female_test) / len(y_female_test))
# p_female = sns.heatmap(cm_female/np.sum(cm_female), cmap="crest", annot=True, fmt='.2%')
# p_female.set_xlabel("Predicted label")
# p_female.set_ylabel("True label")


# ## make MALE X and y TRAIN sets

# In[11]:


#okay actually before I worry about this why don't I just run two different vectorizers/trains on my two arrays...

X_male = []
y_male_train = []

for sentence_arr in male_labeled_array:
    X_male.append(sentence_arr[0])
    y_male_train.append(sentence_arr[1])

tokenizer = RegexpTokenizer(r'\w+')

#df_X_female = pd.DataFrame(X_female)

tokenized_arr = []
for tweet_sentence in X_male:
    tokenized_sentence = tokenizer.tokenize(tweet_sentence.lower())
    tokenized_arr.append(tokenized_sentence)

print(tokenized_arr[0])

list_of_word_arrs = []

for word_arr in tokenized_arr:
    existing_keys_arr = []
    #print(word_arr)
    for word in word_arr:
        if word in glove_twitter_50.key_to_index:
            existing_keys_arr.append(word)
    list_of_word_arrs.append(existing_keys_arr)
    
embed_arr = []
i=0
empty_indices = []
for word_arr in list_of_word_arrs:
    #print(word_arr)
    try:
        embed_arr.append(glove_twitter_50[word_arr])
    except: #TODO: CHANGE THIS
        empty_indices.append(i)
        pass
    i = i + 1
print(empty_indices)

# new_y_female = []

# j = 0
# for element in y_female:
#     if not j in empty_indices:
#         new_y_female.append(element)
#     j = j + 1


male_train_embed_list = []
for sentence_word_embed_list in embed_arr:
    np_sentence_list = np.array(sentence_word_embed_list)
    np_sum_arr = np.sum(np_sentence_list, axis=0)
    sum_arr = np_sum_arr.tolist()
    male_train_embed_list.append(sum_arr)
    
#np_word_embedding = np.array(embed_arr[0])
np_male_train_embed_list = np.array(male_train_embed_list)
print(np_male_train_embed_list.shape)
np_y_male_train = np.array(y_male_train)
print(np_y_male_train.shape)


# ## make MALE X and y TEST sets

# In[12]:


#okay actually before I worry about this why don't I just run two different vectorizers/trains on my two arrays...

X_male = []
y_male_test = []

for sentence_arr in male_test_arr:
    X_male.append(sentence_arr[0])
    y_male_test.append(sentence_arr[1])

tokenizer = RegexpTokenizer(r'\w+')

#df_X_female = pd.DataFrame(X_female)

tokenized_arr = []
for tweet_sentence in X_male:
    tokenized_sentence = tokenizer.tokenize(tweet_sentence.lower())
    tokenized_arr.append(tokenized_sentence)

print(tokenized_arr[0])

list_of_word_arrs = []

for word_arr in tokenized_arr:
    existing_keys_arr = []
    #print(word_arr)
    for word in word_arr:
        if word in glove_twitter_50.key_to_index:
            existing_keys_arr.append(word)
    list_of_word_arrs.append(existing_keys_arr)
    
embed_arr = []
i=0
empty_indices = []
for word_arr in list_of_word_arrs:
    #print(word_arr)
    try:
        embed_arr.append(glove_twitter_50[word_arr])
    except: #TODO: CHANGE THIS
        empty_indices.append(i)
        pass
    i = i + 1
print(empty_indices)

# new_y_female = []

# j = 0
# for element in y_female:
#     if not j in empty_indices:
#         new_y_female.append(element)
#     j = j + 1


male_test_embed_list = []
for sentence_word_embed_list in embed_arr:
    np_sentence_list = np.array(sentence_word_embed_list)
    np_sum_arr = np.sum(np_sentence_list, axis=0)
    sum_arr = np_sum_arr.tolist()
    male_test_embed_list.append(sum_arr)
    
#np_word_embedding = np.array(embed_arr[0])
np_male_test_embed_list = np.array(male_test_embed_list)

np_male_test_embed_list.shape


# In[13]:


#okay so there are three missing values in the embed_train-- these are presumably the values I deleted?
clf = LogisticRegression(max_iter = 2000)
clf.fit(male_train_embed_list, y_male_train)
print("model accuracy on the train:", sum(clf.predict(male_train_embed_list) == y_male_train) / len(y_male_train))


# ## visualize confusion matrix!

# In[14]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# male_test_predictions = clf.predict(male_test_embed_list)
# cm_male = confusion_matrix(y_male_test, male_test_predictions)
# print("TEST accuracy:", sum(clf.predict(male_test_embed_list) == y_male_test) / len(y_male_test))
# p_male = sns.heatmap(cm_male/np.sum(cm_male), cmap="crest", annot=True, fmt='.2%')
# p_male.set_xlabel("Predicted label")
# p_male.set_ylabel("True label")


# ## Train *OVERALL* GENDERED model

# In[15]:


gendered_train_embed_list = []
gendered_test_embed_list = []
y_gendered_train = []
y_gendered_test = []

#For some reason concatenate is not working correctly, so I have to just iterate over the lists

for y_train in y_female_train:
    y_gendered_train.append(y_train)
for y_train in y_male_train:
    y_gendered_train.append(y_train)

for y_test in y_female_test:
    y_gendered_test.append(y_test)
for y_test in y_male_test:
    y_gendered_test.append(y_test)
    
for x_train in female_train_embed_list:
    gendered_train_embed_list.append(x_train)
for x_train in male_train_embed_list:
    gendered_train_embed_list.append(x_train)
    
for x_test in female_test_embed_list:
    gendered_test_embed_list.append(x_test)
for x_test in male_test_embed_list:
    gendered_test_embed_list.append(x_test)

np_x_train = np.array(gendered_train_embed_list)
print(np_x_train.shape)

clf_gendered = LogisticRegression(max_iter = 2000)
clf_gendered.fit(gendered_train_embed_list, y_gendered_train)
print("model accuracy on the train:", sum(clf.predict(gendered_train_embed_list) == y_gendered_train) / len(y_gendered_train))


# In[ ]:





# ## WRONG Make overall gendered plots

# In[ ]:





# ## WRONG Make female plot

# In[16]:


## Female predictions
# female_test_predictions = clf_gendered.predict(female_test_embed_list)
# cm_female = confusion_matrix(y_female_test, female_test_predictions)
# print("TEST accuracy:", sum(clf_gendered.predict(female_test_embed_list) == y_female_test) / len(y_female_test))
# p_female = sns.heatmap(cm_female/np.sum(cm_female), cmap="crest", annot=True, fmt='.2%', xticklabels=["not sexist", "sexist"], yticklabels=["not sexist", "sexist"])
# p_female.set_xlabel("Predicted label")
# p_female.set_ylabel("True label")
# p_female.set_title("Female Gendered")


# ## WRONG Make male plot

# In[17]:


# Male predictions
# male_test_predictions = clf_gendered.predict(male_test_embed_list)
# cm_male = confusion_matrix(y_male_test, male_test_predictions)
# print("TEST accuracy:", sum(clf_gendered.predict(male_test_embed_list) == y_male_test) / len(y_male_test))
# p_male = sns.heatmap(cm_male/np.sum(cm_male), cmap="crest", annot=True, fmt='.2%', xticklabels=["not sexist", "sexist"], yticklabels=["not sexist", "sexist"])
# p_male.set_xlabel("Predicted label")
# p_male.set_ylabel("True label")
# p_male.set_title("Male Gendered")


# ### Extract unigram features

# #### Featurize each tweet in the training set

# In[18]:


from sklearn.feature_extraction.text import CountVectorizer
new_vectorizer = CountVectorizer(ngram_range=(1, 1))
X_kfold = new_vectorizer.fit_transform(small_df['text'])
X_kfold

y_kfold = small_df['label_sexist'].map(lambda x: 0 if "not" in x else 1)
X_kfold.shape


# In[19]:


y_kfold.value_counts(normalize=True)


# In[20]:


y_kfold_arr = np.array(y_kfold)


# ## new tokenizing ...

# In[21]:


tokenizer = RegexpTokenizer(r'\w+')

tokenized_train = train_df['text'].apply(lambda row: tokenizer.tokenize(row.lower()))
tokenized_test = test_df['text'].apply(lambda row: tokenizer.tokenize(row.lower()))

y_train = train_df['label_sexist'].map(lambda x: 0 if "not" in x else 1)
y_test = test_df['label_sexist'].map(lambda x: 0 if "not" in x else 1)

print(tokenized_train)


# ## UNGENDERED word embeddings for TRAIN

# In[22]:


# first, remove elements from each sentence that are not in the embedding matrix
list_of_word_arrs = []

for word_arr in tokenized_train:
    existing_keys_arr = []
    #print(word_arr)
    for word in word_arr:
        if word in glove_twitter_50.key_to_index:
            existing_keys_arr.append(word)
    list_of_word_arrs.append(existing_keys_arr)

embed_arr = []
i=0
empty_indices = []
for word_arr in list_of_word_arrs:
    #print(word_arr)
    try:
        embed_arr.append(glove_twitter_50[word_arr])
    except: #TODO: CHANGE THIS
        empty_indices.append(i)
        pass
    i = i + 1
print(empty_indices)

new_y_train = []

j = 0
for element in y_train:
    if not j in empty_indices:
        new_y_train.append(element)
    j = j + 1

average_embed_list_train = []
for sentence_word_embed_list in embed_arr:
    np_sentence_list = np.array(sentence_word_embed_list)
    np_sum_arr = np.sum(np_sentence_list, axis=0)
    sum_arr = np_sum_arr.tolist()
    average_embed_list_train.append(sum_arr)


# ## UNGENDERED word embeddings for TEST

# In[23]:


# first, remove elements from each sentence that are not in the embedding matrix
list_of_word_arrs = []

for word_arr in tokenized_test:
    existing_keys_arr = []
    #print(word_arr)
    for word in word_arr:
        if word in glove_twitter_50.key_to_index:
            existing_keys_arr.append(word)
    list_of_word_arrs.append(existing_keys_arr)

embed_arr = []
i=0
empty_indices = []
for word_arr in list_of_word_arrs:
    #print(word_arr)
    try:
        embed_arr.append(glove_twitter_50[word_arr])
    except: #TODO: CHANGE THIS
        empty_indices.append(i)
        pass
    i = i + 1
print(empty_indices)

new_y_test = []

j = 0
for element in y_test:
    if not j in empty_indices:
        new_y_test.append(element)
    j = j + 1

average_embed_list_test = []
for sentence_word_embed_list in embed_arr:
    np_sentence_list = np.array(sentence_word_embed_list)
    np_sum_arr = np.sum(np_sentence_list, axis=0)
    sum_arr = np_sum_arr.tolist()
    average_embed_list_test.append(sum_arr)


# ## make bigger dataset (just in case that matters) -- x20

# In[24]:


bigger_list_train = []
bigger_y_train = []
bigger_list_test = []
bigger_y_test = []

for i in range(20):
    bigger_list_train = bigger_list_train + average_embed_list_train
    bigger_list_test = bigger_list_test + average_embed_list_test
    bigger_y_train = bigger_y_train + new_y_train
    bigger_y_test = bigger_y_test + new_y_test

average_embed_list_train = bigger_list_train
new_y_train = bigger_y_train
average_embed_list_test = bigger_list_test
new_y_test = bigger_y_test

print(np.array(average_embed_list_train).shape)
print(np.array(new_y_train).shape)
print(np.array(average_embed_list_test).shape)
print(np.array(new_y_test).shape)


# ## SINGLE SPLIT logistic regression

# In[25]:


np_y_new = np.array(new_y_train)
clf_ungendered = LogisticRegression(max_iter = 2000)
clf_ungendered.fit(average_embed_list_train, new_y_train)
print("model accuracy on the train:", sum(clf_ungendered.predict(average_embed_list_train) == new_y_train) / len(new_y_train))

# from sklearn.neural_network import MLPClassifier
# mlp_ungendered = MLPClassifier(random_state=1, max_iter=100).fit(average_embed_list_train, new_y_train)
# print("model accuracy on the train:", sum(mlp_ungendered.predict(average_embed_list_train) == new_y_train) / len(new_y_train))


# ## Create GENDERED visualizations

# In[26]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

gendered_test_predictions = clf_ungendered.predict(gendered_test_embed_list)
cm_gendered = confusion_matrix(y_gendered_test, gendered_test_predictions)
print("TEST accuracy:", sum(clf_ungendered.predict(gendered_test_embed_list) == y_gendered_test) / len(y_gendered_test))
p_gendered = sns.heatmap(cm_gendered/np.sum(cm_gendered), cmap="crest", annot=True, fmt='.2%', xticklabels=["not sexist", "sexist"], yticklabels=["not sexist", "sexist"])
p_gendered.set_xlabel("Predicted label")
p_gendered.set_ylabel("True label")
p_gendered.set_title("Overall Gendered")


# ## Generate UNGENDERED visualizations

# In[27]:


ungendered_test_predictions = clf_ungendered.predict(average_embed_list_test)
cm_ungendered = confusion_matrix(new_y_test, ungendered_test_predictions)
print("TEST accuracy:", sum(clf_ungendered.predict(average_embed_list_test) == new_y_test) / len(new_y_test), )
p_ungendered = sns.heatmap(cm_ungendered/np.sum(cm_ungendered), cmap="crest", annot=True, fmt='.2%', xticklabels=["not sexist", "sexist"], yticklabels=["not sexist", "sexist"])
p_ungendered.set_xlabel("Predicted label")
p_ungendered.set_ylabel("True label")
p_ungendered.set_title("Base (Ungendered)")


# In[28]:


## NEW Female


# In[29]:


female_test_predictions = clf_ungendered.predict(female_test_embed_list)
cm_female = confusion_matrix(y_female_test, female_test_predictions)
print("TEST accuracy:", sum(clf_ungendered.predict(female_test_embed_list) == y_female_test) / len(y_female_test))


# In[30]:


## NEW Male


# In[31]:


male_test_predictions = clf_ungendered.predict(male_test_embed_list)
cm_male = confusion_matrix(y_male_test, male_test_predictions)
print("TEST accuracy:", sum(clf_ungendered.predict(male_test_embed_list) == y_male_test) / len(y_male_test))


# In[32]:


## Putting confusion matrices side by side


# In[33]:


sns.set(font_scale=1.4)
fig, axs = plt.subplots(ncols=4, figsize=(25,5))
fig.subplots_adjust(hspace=0.125, wspace=0.5)

p_ungendered = sns.heatmap(cm_ungendered/np.sum(cm_ungendered), cmap="crest", annot=True, fmt='.2%', xticklabels=["not sexist", "sexist"], yticklabels=["not sexist", "sexist"], ax=axs[0])
p_ungendered.set_xlabel("Predicted label")
p_ungendered.set_ylabel("True label")
p_ungendered.set_title("Base Test Set")

p_gendered = sns.heatmap(cm_gendered/np.sum(cm_gendered), cmap="crest", annot=True, fmt='.2%', xticklabels=["not sexist", "sexist"], yticklabels=["not sexist", "sexist"], ax=axs[1])
p_gendered.set_xlabel("Predicted label")
p_gendered.set_ylabel("True label")
p_gendered.set_title("Gendered Test Set")

p_male = sns.heatmap(cm_male/np.sum(cm_male), cmap="crest", annot=True, fmt='.2%', xticklabels=["not sexist", "sexist"], yticklabels=["not sexist", "sexist"], ax=axs[2])
p_male.set_xlabel("Predicted label")
p_male.set_ylabel("True label")
p_male.set_title("Male Test Set")

p_female = sns.heatmap(cm_female/np.sum(cm_female), cmap="crest", annot=True, fmt='.2%', xticklabels=["not sexist", "sexist"], yticklabels=["not sexist", "sexist"], ax=axs[3])
p_female.set_xlabel("Predicted label")
p_female.set_ylabel("True label")
p_female.set_title("Female Test Set")



# ## Calculate the f1 scores

# In[34]:


from sklearn.metrics import f1_score

sns.set(font_scale=1.2)

#TODO: add precision and recall

ungendered_f1 = f1_score(new_y_test, ungendered_test_predictions)
gendered_f1 = f1_score(y_gendered_test, gendered_test_predictions)
male_f1 = f1_score(y_male_test, male_test_predictions)
female_f1 = f1_score(y_female_test, female_test_predictions)

print("ungendered F1 ", ungendered_f1)
print("gendered F1 ", gendered_f1)
print("male F1 ", male_f1)
print("female F1 ", female_f1)

x = ["Base", "Gendered", "Male", "Female"]
y = [ungendered_f1, gendered_f1, male_f1, female_f1]

f1_df = pd.DataFrame(data={'model':x,'f1':y})

ax = f1_df.plot.bar(x='model', y='f1', rot=0)
bar = sns.barplot(data=f1_df, x='model', y='f1')
bar.set_xlabel("Test Set")
bar.set_ylabel("F1-Score")

bar.legend([], frameon=False)


# ## Calculate predicted probability differences

# In[35]:


missing_indices = [4023, 5610, 6844, 8320, 12170]

ungendered_prob = clf_ungendered.predict_proba(average_embed_list_test)
male_prob = clf_ungendered.predict_proba(male_test_embed_list)
female_prob = clf_ungendered.predict_proba(female_test_embed_list)

np_ungendered_prob = np.array(ungendered_prob)
np_male_prob = np.array(male_prob)
np_female_prob = np.array(female_prob)

half_ungendered_prob, other_half = np.split(np_ungendered_prob, 2)
print(half_ungendered_prob.size)
print(np_male_prob.size)
print(np_female_prob.size)

male_dif = np_male_prob - half_ungendered_prob
female_dif = np_female_prob - half_ungendered_prob

#need to check this??
male_sexist_dif = male_dif[0:,1]
female_sexist_dif = female_dif[0:,1]

print(half_ungendered_prob[0])
print(np_male_prob[0])

average_male_dif = sum(male_sexist_dif)/len(male_sexist_dif)
average_female_dif = sum(female_sexist_dif)/len(female_sexist_dif)

print(average_male_dif)
print(average_female_dif)

plot1 = sns.histplot(data=[male_sexist_dif, female_sexist_dif], binwidth=0.1)
#plot2 = sns.histplot(data=female_sexist_dif, binwidth=0.1)
plot1.legend(["male", "female"])

plot1.set_xlabel("Difference in prob. 'sexist' relative to base")


# ## Looking at weights

# In[36]:


clf.classes_


# ### Which feature had the strongest weight

# In[37]:


# this is the index of that feature
np.argmax(clf.coef_)


# In[38]:


np.argsort(clf.coef_)[0][-10:]


# In[39]:


# this is the name of the feature
vectorizer.get_feature_names()[np.argmax(clf.coef_)]


# In[ ]:


for i in np.argsort(clf.coef_)[0][-10:]:
    print(vectorizer.get_feature_names()[i])


# In[ ]:


help(np.argmax)

