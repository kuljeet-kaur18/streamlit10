#!/usr/bin/env python
# coding: utf-8

# ## Firstly we need to import the following libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Now we need to run the following code to read the csv file.

# In[3]:


data= pd.read_csv("C:\\Users\\kulje\\Downloads\\bank-additional-full (1).csv") #separator=; as per given csv file
data.head()


# ## For the information of the data.

# In[4]:


data.info()


# ### Now we will rename the target column or variable 'y' as 'term_deposit'.

# In[5]:


data = data.rename(columns = {'y' : 'term_deposit'})

data.info()


# In[6]:


#check descriptive statistics 
data.describe().T


# **Observation:** Age ranges from 18 to 95. We have duration max for a 4918 seconds. We have balance ranges from -8019 to 102127.

# ### We will do the appropriate conversion to numeric data before passing to our ML Models.

# In[7]:


print(f'Data contains {data.shape[0]} samples and {data.shape[1]} variables')

#ID_col=
TARGET_COL='term_deposit'

#features = [c for c in data.columns if c not in [ID_COL, TARGET_COL]]
features = [c for c in data.columns if c not in [TARGET_COL]]
print(f'\nThe dataset contains {len(features)} input features')


# ### Distribution of the Target Feature: 'term_deposit' - has the client subscribed a term deposit? (binary: 'yes', 'no')

# In[8]:


data[TARGET_COL].value_counts(normalize='True') #normalise='True' returns % according to frequency


# In[9]:


sns.countplot(data[TARGET_COL])
plt.title("Target Distribution", fontsize=14)


# **Observation:** It can be seen from above graph that It is a highly imbalanced dataset which was expected as with every marketing campaign.

# ## Data Cleaning and EDA

# In[10]:


#Check for Null Values
data[features].isnull().sum()


# **Observation:** There are no null values in any column.

# In[11]:


#removal of duplicate rows
print(data.shape)
data = data.drop_duplicates()
print(data.shape)


# **Observation:** There were some duplicate values which we have removed in the above part.

# In[12]:


#Checking Unique values in each variable
data.nunique()      # nunique() method returns the no. of unique values for each column


# **Observation:** We can see the features 'age', 'duration', 'campaign' and 'pdays' contain a lot of unique values.

# ## Analyzing Each Variable & their Relationships
# 
# ## For each feature type we will perform two types of analysis:
# 
# ### Univariate: Analyze 1 feature at a time
# 
# ### Bivariate: Analyze the relationship of that feature with target variable, 'term_deposit'

# In[13]:


data.columns


# In[14]:


#Segregating Categorical & Numerical Variables
cat_cols = ['job','marital','education','default','housing','loan','contact','month','day_of_week', 'poutcome']

num_cols = [c for c in features if c not in cat_cols]
num_cols


# In[15]:


#Univariate Analysis of Categorical Features

#A way of looking at the counts of each of the categories is countplots. 
#These are similar to barplots with the width of the bars representing the count of the category in the variable.

fig, axes = plt.subplots(5, 2, figsize=(16, 16)) #total 10 subplots that's why 5*2
axes = [ax for axes_row in axes for ax in axes_row]

for i, c in enumerate(data[cat_cols]):
    _ = data[c].value_counts()[::-1].plot(kind = 'barh', ax=axes[i], title=c, fontsize=14)
    
_ = plt.tight_layout()


# In[16]:


#Bivariate Analysis: Relationships of Categorical Features with Target


for c in cat_cols:
    plt.figure(figsize=(18,6)) #this creates a new figure on which your plot will appear
    sns.countplot(x =c, hue='term_deposit',data = data,order = data[c].value_counts().index) #individual bars sorted acc to frequency and then plotted


# **Observation:** From the day_of_week plot, we can see that, all the days have the similar distribution for both the classes. Thus we won't be using it in predicting the target variable. We have dropped this feature before making our models below!

# In[17]:


#Numerical Features


#Univariate Analysis using Histograms
data.hist(color = "k",
        bins = 30,
        figsize = (20, 15))
plt.show()


# **Observation:** We can see some features are skewed and not normally distributed. The never contacted before respondents skew the variables “campaign” and “previous” towards zero. But it is not compulsary to make the features normally distributed, 
# so we won't do that right now.
# 
# The histogram of pdays looks so strange because most records have pdays as 999. The distribution of the number of days 
# since the previous campaign (“pdays”) is skewed towards 1,000 because, for the respondents who were never contacted, 
# the value is 999. The other values of pdays are very small comaparitively.
# 
# We have handled this feature in code below before building our models.

# In[18]:


fig, axes = plt.subplots(10, 1, figsize=(8, 60))
for i, c in enumerate(num_cols):
    sns.boxplot(data=data,x='term_deposit',y=c,ax=axes[i])  


# **Observation:** The boxplots for both the classes overlap quite a lot, which means that those particular features aren't necessarily a good indicator for which customer will subscribe and which customer will not.
# 
# But features like 'emp.var.rate' and 'euriborm3m' seem very useful as we can clearly see the difference in median for both 
# the classes according to these features.

# In[19]:


#Univariate Analysis using Density Plots

#A kernel density estimate (KDE) plot is a method for visualizing the distribution of observations in a dataset, analagous to a histogram. 
#KDE represents the data using a continuous probability density curve in one or more dimensions.

sns.set(font_scale=1.3)
fig, axes = plt.subplots(5, 2, figsize=(18, 14))
axes = [ax for axes_row in axes for ax in axes_row]
for i, c in enumerate(num_cols):
    plot = sns.kdeplot(data=data, x=c, ax=axes[i], fill=True)
plt.tight_layout()


# We could confirm our observation about 'pdays' feature from its kde plot as well!

# In[20]:


# Bivariate Analysis KDE plots - Relationships with Target Variable

sns.set(font_scale=1.3)
fig, axes = plt.subplots(5, 2, figsize=(18, 14))
axes = [ax for axes_row in axes for ax in axes_row]
for i, c in enumerate(num_cols):
    plot = sns.kdeplot(data=data, x=c, hue=TARGET_COL, multiple='fill', ax=axes[i])
plt.tight_layout()


# In[21]:


#Bivariate Analysis - Correlation Heatmaps

plt.figure(figsize=(14, 8))
_ = sns.heatmap(data[num_cols].corr(), annot=True)


# **Observation:** So, from this correlation heatmap, we can see that how the different variables are correlated with each other. Here we can see that age and duration are highly negatively correlated with each other, similarly campaign and cons.price.idx with previous, pdays with cons.conf.idx, emp.var.rate with age, euribor3m and and nr.employed with duration. And every variable has only positive correlation with each other.

# In[22]:


data1= data.copy()

data1['term_deposit'] = data1['term_deposit'].replace(['no','yes'],[0,1]) #needed to do below mathematical operations
data1.head()

#no of clients who subscribed to the term deposit grouped by occupation
total_subscribers = data1.groupby('job').term_deposit.sum().values

# Proportion of clients who subscribed to the term deposit grouped by occupation
proportion_subscribed = (round(data1.groupby('job').term_deposit.sum()/data1.groupby('job').term_deposit.count(),3)*100).values

# Total amount of clients per occupation
total_people = data1.groupby('job').term_deposit.count().values

#Form a dataframe and print
jobs = sorted(data1.job.unique()) #list of all jobs
jobs_with_subscribers = pd.DataFrame({'Job': jobs, 'Total Subscribers':total_subscribers,'Total People in Job': total_people,'Proportion of Subscribers': proportion_subscribed})
jobs_with_subscribers.sort_values(by='Proportion of Subscribers', ascending=False)


# **Observation:** Though the number of admin, blue collar, technician subscribers are more, we can also see that according to proportion, students and retired people are much more likely to subscribe to our term deposits!

# In[23]:


married_subscribers = data1[(data1.marital=='married') ].term_deposit.sum()
single_subscribers = data1[(data1.marital=='single') | (data1.marital=='divorced')].term_deposit.sum() #single or divorced

married_subscribers_prop = married_subscribers/len(data1[data1.marital=='married'])
single_subscribers_prop = single_subscribers/len(data1[(data1.marital=='single') | (data1.marital=='divorced')])

print('No of Married clients who subscribe: {}'.format(married_subscribers))
print('No of Single (and divorced) clients who subscribe : {}'.format(single_subscribers))

print('Married clients campaign success rate: {0:.0f}%'.format(married_subscribers_prop*100))
print('Single clients campaign success rate: {0:.0f}%'.format(single_subscribers_prop*100))


# **Observation:** Even though we have more clients who are married subscribers, If we look according to proportions, we see that single clients responded to the campaign better.

# In[24]:


n1=len(data[(data['age'] > 60) & (data['term_deposit'] == 'yes')]) #number of old subscribers
n2=len(data[(data['age'] > 60) & (data['term_deposit'] == 'no')]) #number of old non-subscribers
n3=len(data[(data['age'] <= 60) & (data['term_deposit'] == 'yes')]) #number of young subscribers
n4=len(data[(data['age'] <= 60) & (data['term_deposit'] == 'no')]) #number of young non-subscribers

print(f'Proportion of young subscribers is: {n3/(n3+n4)} and the proportion of old subscribers is {n1/(n1+n2)}')


# **Observation:** It looks like effect of marketing on old people is much more positive than corresponding effect on young people! So old people can be a major group we would like to target!

# ### Answering Hypothesis Questions:
# 
# #### 1.Are senior (retired) people more likely to subscribe for term deposits? (they may prefer safe investments)
# 
# Ans- No. The number of subscriptions by retired people are less. But We can see a positive result of the marketing campaign on senior people i.e the proprtion of yes/no is more for retired folks.
# 
# Similarly, it is also interesting to see a very positive effect of the marketing campaign on students.
# 
# #### 2.Do salaried people prefer it more than business owners who would invest money into their business rather than putting in bank?
# 
# Ans- Yes. Salaried People (admin, service, technician, blue collar jobs etc. ) are much more interested than entrepreneurs and self employed people.
# 
# #### 3.Are married people more likely to subscibe for term deposits? (They may prefer having savings for their children?)
# 
# Ans- No, Proportion of single subscribers is more by 3%.
# 
# #### 4.If you already have loans, would you be less likely to subscibe to term deposits?
# 
# Ans- People having personal loans subscribe less to the term deposits. The same is not true for people with home loans.
# 
# #### 5.Are younger customers more likely to subscribe to a term deposit compared to old customers ?
# 
# Ans- It's not very clear that younger customers are more likely to subscribe to a term deposit compared to old customers but by comparing the number of subscriptions by both, we can say that old people can be targeted more for convincing them to subscribe for the term deposits.

# ## Feature Engineering and Data Preprocessing

# In[25]:


#As given in dataset description, we won't use 'duration' column. 
#Reason: We should note here the column “duration” has an interesting relationship with the target variable. 
#If there were no phone calls (i.e. duration =0) then our target, y, has to be no. This will cause some unbalancing in the model and would inhibit the predictive power. In order to have a more realistic prediction, we will drop this column from our dataframe.

#We can't predict how long we gonna talk to the client (duration) and 
#how many calls would require to get the answer about deposit (campaign), so let's drop these! 

data = data.drop(['duration','campaign'],axis=1)
data.head()


# In[26]:


#replacing no and yes by 0 and 1 respectively as our target values!

data['term_deposit'] = data['term_deposit'].replace(['no','yes'],[0,1])
data.head()


# In[27]:


#let's remove these two categorical features which we think won't be useful in building our ML models

data.drop(['month','day_of_week'],axis=1,inplace=True)
data.head()


# In[28]:


#Converting all Categorical Variables to numbers.

new_cat_cols=['job','marital','education','default','housing','loan','contact','poutcome']

for c in new_cat_cols:
  print(data[c].value_counts())


# In[29]:


#dropping rows with 'unknown' values for any categorical column.

index_names= data[(data.job == 'unknown') | (data.marital == 'unknown') | (data.education == 'unknown') | (data.default == 'unknown') | (data.housing == 'unknown') | (data.loan == 'unknown')].index
#print(index_names)
  
# drop these given row 
# indexes from dataFrame 
data.drop(index_names, inplace = True) 
data.shape


# In[30]:


new_cat_cols=['job','marital','education','default','housing','loan','contact','poutcome']
for c in new_cat_cols:
  print(data[c].value_counts())


# **Observation:** Since number of categories for each column is low, we can use ONE HOT ENCODING.

# In[31]:


data = pd.get_dummies(data, columns=new_cat_cols)
data.head()


# In[32]:


data['pdays'].value_counts()


# **Observation:** As we can see most records have pdays as 999. This means most records indicate the particular person was not contacted before. Let's convert this pdays column into a binary categorical column with 2 values: 0: pdays is 999 i.e person was not contacted and 1: pdays!=999 i.e person was contacted before.

# In[33]:


#data[new_column]=np.where(condition, value if condition is true, value if condition is false)

data['has_contacted'] = np.where(data['pdays']!= 999, True, False)

data.drop(['pdays'],axis=1,inplace=True)

data.head()


# In[34]:


data = pd.get_dummies(data,columns=['has_contacted'])
data.head()


# In[35]:


features = [c for c in data.columns if c not in [TARGET_COL]]
len(features)


# **Splitting the combined dataset after preprocessing into train and test sets.**
# 
# We will use 80-20 split with 80% of the rows belonging to training data. 
# 
# Stratified Sampling is necessary, since the dataset is highly imbalanced. **Stratified sampling ensures that the minority class is distributed proportionally among the two classes.**

# In[36]:


from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.2, random_state = 1, stratify = data[TARGET_COL]) #stratified sampling
train.shape, test.shape


# In[37]:


train.head()


# In[38]:


#Input to our model will be the features
X_train, X_test = train[features], test[features]

#Output of our model will be the TARGET_COL
y_train, y_test = train[TARGET_COL], test[TARGET_COL]


# ## Modelling

# #### Performance Metrics for our Models:
# A classifier is only as good as the metric used to evaluate it. If we choose the wrong metric to evaluate our models, we are likely to choose a poor model, or in the worst case, be misled about the expected performance of your model.
# 
# Classification Accuracy should not be used as a metric for imbalanced classification. This is so because even if our model is not intelligent and just guesses all clients as the majority class "not subscribing to the term deposit", we will get a very high accuracy.
# 
# When we want to give equal weight to both classes prediction ability we should look at the ROC curve. ROC Area Under Curve (AUC) Score is used as the metric for imbalanced data. ROC AUC score gets over the above described problem by looking into both the True positive rate (TPR) and False positive rate (FPR). Only if both the TPR and FPR are well above the random line in the ROC curve, we will get a good AUC. Accuracy does not guarantee that.
# 
# We will also be seeing F1-Score as our secondary performance metric to analyze the performance of our models. What we are trying to achieve with the F1-score metric is to find an equal balance between precision and recall, which is extremely useful in most scenarios when we are working with imbalanced datasets.

# ### Logistic Regression Model

# In[39]:


#Standardize features (mainly numeric) by removing the mean and scaling to unit variance. 
#This is necessary for Logistic Regression.
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
_ = scaler.fit(X_train)
X_train = scaler.transform(X_train)
#X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# In[40]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

lr = LogisticRegression(max_iter=7600,random_state = 1)
_ = lr.fit(X_train, y_train)

#predictions on test data
preds_test= lr.predict(X_test)

#f1 score on test set
f1_score(y_test, preds_test)


# In[41]:


from sklearn.metrics import roc_auc_score,roc_curve,auc,confusion_matrix,classification_report

print("Confusion Matrix is:")
print(confusion_matrix(y_test, preds_test))


# In[42]:


pd.crosstab(y_test, preds_test, rownames=['Actual'], colnames=['Predicted'], margins=True)


# In[43]:


print("AUC on Test data is " +str(roc_auc_score(y_test,preds_test)))


# ### Decision Tree Classifier

# In[44]:


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state = 1)
_ = dt.fit(X_train, y_train)

#predictions on test data
preds_test= dt.predict(X_test)

#f1 score on test set
f1_score(y_test, preds_test)


# We need to do hyperparameter tuning to improve our performance. Hyper-parameters and their values vary from dataset to dataset, and their optimal values have a large impact on the performance of our model.

# #### Random Search for Hyperparameter Tuning
# 
# In random search we will run our model only a fixed number of times, say 10, and among these 10 runs we will return the best hyper-parameter combination. This may not be the optimal hyper-parameter combination. But, it saves much more time than Grid Search, so we will go for random search.

# In[45]:


from sklearn.model_selection import RandomizedSearchCV

hyperparam_combs = {
    'max_depth': [4, 6, 8, 10, 12],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [2, 10, 20, 30, 40],
    'max_features': [0.2, 0.4, 0.6, 0.8, 1],
    'max_leaf_nodes': [8, 16, 32, 64, 128],
    'class_weight': [{0: 1, 1: 1}, {0: 1, 1: 2}, {0: 1, 1: 3}, {0: 1, 1: 4}, {0: 1, 1: 5}]
}

dt2 = RandomizedSearchCV(DecisionTreeClassifier(),
                         hyperparam_combs,
                         scoring='f1',
                         random_state=1,
                         n_iter=30)

search = dt2.fit(X_train, y_train)

search.best_params_


# In[46]:


optimal_params = {
 'criterion': 'entropy',
 'max_depth': 10,
 'max_features': 0.6,
 'max_leaf_nodes': 32,
 'min_samples_split': 20,
 'class_weight': {0: 1, 1: 3}}

dt2 = DecisionTreeClassifier(random_state = 1, **optimal_params)
_ = dt2.fit(X_train, y_train)


#predictions on test data
preds_test= dt2.predict(X_test)

#f1 score on test set
f1_score(y_test, preds_test)


# **0.528**, So our performance has increased after hyperparameter tuning.

# In[47]:


print("Confusion Matrix is:")
print(confusion_matrix(y_test, preds_test))


# In[48]:


pd.crosstab(y_test, preds_test, rownames=['Actual'], colnames=['Predicted'], margins=True)


# In[49]:


print("AUC on Test data is " +str(roc_auc_score(y_test,preds_test)))


# ### Random Forrest Classifier

# In[50]:


#creation of random forrest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=1) #by default no of estimators=10
rf.fit(X_train, y_train)

#predictions on test data
preds_test= rf.predict(X_test)

#f1 score on test set
f1_score(y_test, preds_test)


# In[51]:


# Look at parameters used by our current forest
print('Parameters currently in use:\n')
print(rf.get_params())


# In[52]:


from sklearn.model_selection import RandomizedSearchCV

hyperparam_combs = {
    'max_depth': [4, 6, 8, 10, 12],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [2, 10, 20, 30, 40],
    'max_features': [0.2, 0.4, 0.6, 0.8, 1],
    'max_leaf_nodes': [8, 16, 32, 64, 128],
    'class_weight': [{0: 1, 1: 1}, {0: 1, 1: 2}, {0: 1, 1: 3}, {0: 1, 1: 4}, {0: 1, 1: 5}]
}

rf2 = RandomizedSearchCV(RandomForestClassifier(),
                         hyperparam_combs,
                         scoring='f1',
                         random_state=1,
                         n_iter=30)

search = rf2.fit(X_train, y_train)

search.best_params_


# In[53]:


optimal_params = {
 'criterion': 'gini',
 'max_depth': 10,
 'max_features': 0.2,
 'max_leaf_nodes': 128,
 'min_samples_split': 10,
 'class_weight': {0: 1, 1: 5}}

rf2 = RandomForestClassifier(random_state = 1, **optimal_params)
_ = rf2.fit(X_train, y_train)

#predictions on test data
preds_test= rf2.predict(X_test)

#f1 score on test set
f1_score(y_test, preds_test)


# In[54]:


print("Confusion Matrix is:")
print(confusion_matrix(y_test, preds_test))


# In[55]:


pd.crosstab(y_test, preds_test, rownames=['Actual'], colnames=['Predicted'], margins=True)


# In[56]:


print("AUC on Test data is " +str(roc_auc_score(y_test,preds_test)))


# The performance is slightly improved over the Decision Tree Classifier.

# In[57]:


#predictions on train data
preds_train= rf2.predict(X_train)

#f1 score on train set
f1_score(y_train, preds_train)


# In[58]:


#Auc on Train Data
print("AUC on Train data is " +str(roc_auc_score(y_train,preds_train)))


# Thus we also checked that our Random Forrest model is not overfitting to train data!

# In[60]:


#Visualizing Feature Importance
def plot_feature_importances(model):
    n_features = X_train.shape[1]
    plt.figure(figsize=(12,12))
    plt.barh(range(n_features), model.feature_importances_, align='center') 
    plt.yticks(np.arange(n_features), train[features].columns.values) #trn.columns has list of all columns in our training data
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")

plot_feature_importances(rf2)    


# From the graph we can see that numeric features had the maximum feature imprtance.

# Conclusion:
# 
# In this project, we learned how to utilize Machine Learning to predict if a customer will subscribe to a bank's term deposit
# scheme through its marketing campaign. We found that tree based models like Decision Tree and Random Forrest are giving a
# good performance on this dataset. This is explainable as usually tree based models perform well 
# when number of features are not that large. The best performing model was the hyperparameter tuned 
# Random Forrest Model with F1 score of 0.528, and ROC_AUC score of 0.763 on the test dataset.

# In[ ]:




