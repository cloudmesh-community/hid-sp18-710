
# coding: utf-8

# GUN VIOLENCE IN AMERICA
# Dataset : https://data.world/azel/gun-deaths-in-america

# IMPORT LIBRARIES

# In[1]:


import numpy as np 
import pandas as pd 
from io import BytesIO
import six
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import preprocessing 
from sklearn import model_selection
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
import seaborn as sns 
import pydotplus
from IPython.display import Image
from fbprophet import Prophet
import graphviz
import calendar
import datetime
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics


# READ THE INPUTFILE

# In[2]:


#load dataset
file_location ='C:/Users/gauth/downloads/full_data.csv'
gv_dataset = pd.read_csv(file_location)

# remove rows with NA 
gv_dataset = gv_dataset.dropna()
gv_dataset = gv_dataset.drop(['Unnamed: 0'], axis=1)

gv_dataset.head(5)


# UNDERSTAND THE DATA

# In[3]:


gv_dataset.info()
gv_dataset.describe()


# In[4]:


gv_dataset.notnull().sum()
gv_dataset.notnull().sum() * 100.0/gv_dataset.shape[0]
gv_dataset.head(10)


# FUNCTION TO CONVERT COUNT TO PERCENTAGE

# In[5]:


def val_count_to_percent(column): 
    return pd.value_counts(column)/(pd.value_counts(column).sum())*100


# DATA ANALYSIS

# In[6]:


pd.value_counts(gv_dataset['race'])


# In[7]:


val_count_to_percent(gv_dataset['race'])


# PLOT THE DATASET BY COLUMN RACE

# In[8]:


column  = gv_dataset['race']
fig_width = 15 
fig_height = 15
figname = 'race'

height = np.array(val_count_to_percent(column).values)

# sns.plt.figure(figsize=(fig_width,fig_height))
hue = list(val_count_to_percent(column).index)
mod_hue =[hue[0],hue[1],hue[2],'A/PI', 'NA/NAL'  ]
all_data = {'Feature labels':mod_hue , 'Percent of data': height, 'Race': hue}
df = pd.DataFrame(data = all_data)

ax = sns.catplot(x ='Feature labels', y='Percent of data',hue= 'Race',kind="bar",data =df, legend_out = True, height= 7, aspect = 1.5)
sns.set(font_scale=2.2)



# PLOT THE DATASET BY COLUMN INTENT

# In[9]:


print(val_count_to_percent(gv_dataset['intent']))

column  = gv_dataset['intent']
fig_width = 15 
fig_height = 15
figname = 'intent'

height = np.array(val_count_to_percent(column).values)
hue = list(val_count_to_percent(column).index)
all_data = {'Feature labels':hue , 'Percent of data': height, 'Intent': hue}
df = pd.DataFrame(data = all_data)

ax = sns.catplot(x ='Feature labels', y='Percent of data',hue= 'Intent', kind = 'bar', data =df, legend_out = False, height= 9, aspect = 1.5)



# PLOT THE DATASET BY COLUMN SEX

# In[10]:


print(val_count_to_percent(gv_dataset['sex']))

column  = gv_dataset['sex']
fig_width = 10
fig_height = 10
figname = 'Gender'

height = np.array(val_count_to_percent(column).values)

hue = list(val_count_to_percent(column).index)
mod_hue=['Male(M)','Female(F)']
all_data = {'Feature labels':hue , 'Percent of data': height, 'Gender': mod_hue}
df = pd.DataFrame(data = all_data)

ax = sns.catplot(x ='Feature labels', y='Percent of data',hue= 'Gender', kind = 'bar', data =df, legend_out = False, height= 7, aspect = 1.5)
sns.set(font_scale=2.2)


# PLOT THE DATASET BY COLUMN YEAR

# In[11]:


print(val_count_to_percent(gv_dataset['year']))

column  = gv_dataset['year']
fig_width = 10
fig_height = 10
figname = 'year'

height = np.array(val_count_to_percent(column).values)

# sns.plt.figure(figsize=(fig_width,fig_height))
hue = list(val_count_to_percent(column).index)
mod_hue=hue
all_data = {'Feature labels':hue , 'Percent of data': height, 'Year': mod_hue}
df = pd.DataFrame(data = all_data)

ax = sns.catplot(x ='Feature labels', y='Percent of data',hue= 'Year', kind = 'bar', data =df, legend_out = True,height= 7, aspect = 1.5)
sns.set(font_scale=2.2)


# REPLACE AND VISUALIZE CATEGORICAL COLUMN WITH NUMBERS

# In[12]:


columns_to_encode =  ['intent','sex','place','education']
le = preprocessing.LabelEncoder()

for i  in range(len(columns_to_encode)): 
    column =columns_to_encode[i]
    gv_dataset[column] = le.fit_transform(gv_dataset[column])


# In[13]:


print(val_count_to_percent(gv_dataset['intent']))
print(val_count_to_percent(gv_dataset['police']))


# In[14]:


print(val_count_to_percent(gv_dataset['place']))

column  = gv_dataset['place']
fig_width = 18 
fig_height = 10

height = np.array(val_count_to_percent(column).values)
hue = list(val_count_to_percent(column).index)
mod_hue=['Home', 'OS','Street', 'OU', 'T/SA', 'S/I', 'Farm', 'I/C', 'RI', 'Sports']
all_data = {'Location':mod_hue , 'Percent of data': height, 'Place': hue}
#all_data = {'Place':mod_hue , 'Percent of data': height}
df = pd.DataFrame(data = all_data)

ax = sns.catplot(x ='Location', y='Percent of data',hue='Place', kind = 'bar', data =df, legend_out = True, height= 7, aspect = 1.5)
sns.set(font_scale=2.2)


# DROP THE COLUMNS AND ROWS THAT ARE NOT NEEDED FOR THE MACHINE LEARNING MODEL

# In[15]:


gv_dataset = gv_dataset.drop(['police'], axis=1)


# In[16]:


drop_rows = np.where((gv_dataset['race']=='Hispanic') | (gv_dataset['race']=='Asian/Pacific Islander')
                                    |(gv_dataset['race']=='Asian/Pacific Islander') 
         | (gv_dataset['race']=='Native American/Native Alaskan') )


# In[17]:


gv_dataset.drop(gv_dataset.index[list(drop_rows[0])], inplace=True)


# In[18]:


if 'hispanic' in gv_dataset.columns:
    gv_dataset = gv_dataset.drop(['hispanic'], axis=1)


# In[19]:


np.unique(gv_dataset['race'])


# In[20]:


gv_dataset['race'] = le.fit_transform(gv_dataset['race'])
gv_dataset.head()


# CREATING TRAINING AND TESTING DATASET

# In[21]:


train_X, test_X = model_selection.train_test_split(gv_dataset,test_size=0.3,random_state= 1)


# In[23]:


train_y = train_X['race']
train_X = train_X.drop(['race'], axis=1)

test_y = test_X['race']
test_X = test_X.drop(['race'], axis=1)


# DECISION TREE ALGORITHM

# In[24]:


clf = tree.DecisionTreeClassifier(criterion='gini', max_depth = 8,
                                  min_samples_leaf = 100)

clf = clf.fit(train_X, train_y)
y_preds = clf.predict(test_X)


# In[25]:


print('accuracy: ', metrics.accuracy_score(test_y,y_preds))


# In[26]:


print('recall: ', metrics.recall_score(test_y,y_preds))   
print('precision: ', metrics.precision_score(test_y,y_preds))


# In[27]:


important_features = clf.feature_importances_

#column names 
column_names = train_X.columns
fig_width = 18 
fig_height = 10
hue = column_names
mod_hue=hue
all_data = {'Feature labels':hue, 'Importance score': important_features, 'Feature importances': mod_hue}
df = pd.DataFrame(data = all_data)

ax = sns.catplot(x ='Feature labels', y='Importance score',hue= 'Feature importances', kind = 'bar', data =df, legend_out = False, height= 7, aspect = 1.5)
sns.set(font_scale=2.2)
ax.set(ylim= (0,1))


# In[28]:


# dot_data = StringIO()
# export_graphviz(clf, out_file=dot_data,  
#                 filled=True, rounded=True,
#                 special_characters=True)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
# img_object = Image(graph.write_png('decisiontree.png'))
# Image(graph.create_png())


# RANDOM FOREST ALGORITHM

# In[29]:


# RandomForestClassifier
#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(train_X,train_y)

y_pred=clf.predict(test_X)


# In[30]:


print("Accuracy:",metrics.accuracy_score(test_y, y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(test_y, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(test_y, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_y, y_pred)))  


# KNN ALGORITHM

# In[31]:


from sklearn.neighbors import KNeighborsClassifier

#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)

#Train the model using the training sets
knn.fit(train_X, train_y)

#Predict the response for test dataset
y_pred = knn.predict(test_X)


# In[32]:


print("Accuracy:",metrics.accuracy_score(test_y, y_pred))


# FORECASTING USING FBPROPHET

# In[33]:


# Add date column using year & month & date as 01
gv_dataset['date'] = pd.to_datetime((gv_dataset.year * 10000 + gv_dataset.month * 100 + 1).apply(str),format='%Y%m%d')
#gv_dataset.dtypes.tail(1)
del gv_dataset['year']
del gv_dataset['month']


# In[34]:


monthly_rates = pd.DataFrame(gv_dataset.groupby('date').size(), columns=['count'])
df = gv_dataset.groupby('date').date.agg('count').to_frame('count').reset_index()

df = df.rename(columns={'date': 'ds',
                        'count': 'y'})


# In[35]:


sns.set(font_scale=1) 
ax = df.set_index('ds').plot(figsize=(12, 8))

ax.set_ylabel('Incident Count')
ax.set_xlabel('Date')

plt.show()


# In[36]:


model = Prophet()
model.fit(df)


# In[37]:


future_dates = model.make_future_dataframe(periods=5,freq='Y')
forecast = model.predict(future_dates)


# In[38]:


forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[39]:


forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()


# In[40]:


plot1 = model.plot(forecast)


# In[41]:


model.plot_components(forecast)


# CROSS VALIDATION

# In[42]:


df_cv = cross_validation(model, initial='730 days', period='180 days', horizon = '365 days')
df_cv.head()


# OBTAINING THE PERFORMANCE METRICS

# In[43]:


df_p = performance_metrics(df_cv)
df_p.head()

