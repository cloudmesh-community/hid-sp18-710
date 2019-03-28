
# coding: utf-8

# # GUN VIOLENCE IN AMERICA
# 
# Dataset : http://www.gunviolencearchive.org/

# IMPORT LIBRARIES

# In[2]:


import pandas as pd 
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns 
color = sns.color_palette()
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
import plotly.tools as tls
from numpy import array
from matplotlib import cm

from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import datetime as dt
import warnings
import string
import time
punctuation = string.punctuation
import cufflinks as cf
cf.go_offline()


# READ THE INPUT FILE

# In[3]:


df = pd.read_csv('C:/Users/gauth/Downloads/Gun.csv', index_col=0, parse_dates=[1])


# UNDERSTAND THE DATA

# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# DATA PREPARATION

# In[7]:


df.index.name = 'Index'
df.head()
df.columns = map(str.capitalize, df.columns)
df.columns
df.dtypes


# In[8]:


df.notnull().sum()
df.notnull().sum() * 100.0/df.shape[0]
df.sort_values(['Date'], inplace=True)
df.head(10)


# In[9]:


total = df.isnull().sum().sort_values(ascending = False)
percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
missing_gun_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_gun_data.head(10)


# In[10]:


import calendar
monthly_rates = pd.DataFrame(df.groupby('Date').size(), columns=['Count'])
monthly_rates.index.to_datetime
print(monthly_rates.index.dtype)
print(monthly_rates.shape)

days_per_month = []
for val in monthly_rates.index:
    days_per_month.append(calendar.monthrange(val.year, val.month)[1])
monthly_rates['Days_per_month'] = days_per_month
monthly_rates

df['date'] = pd.to_datetime(df['Date'])
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['monthday'] = df['Date'].dt.day
df['weekday'] = df['Date'].dt.weekday
df['loss'] = df['N_killed'] + df['N_injured']


# Analysing Gun Violence Trends based on time factors

# Time Series Analysis

# In[11]:


df['Total_Number_Of_Affected_People'] = df['N_killed'] + df['N_injured']
cols_to_use = ['Date','N_killed', 'N_injured', 'Total_Number_Of_Affected_People']
temp = df[cols_to_use]
temp = temp.groupby('Date').sum()
temp = temp.reset_index()

year2013_to_2018 = temp[temp.Date.dt.year.isin([2013]) | temp.Date.dt.year.isin([2014]) | temp.Date.dt.year.isin([2015]) | temp.Date.dt.year.isin([2016]) | temp.Date.dt.year.isin([2017]) | temp.Date.dt.year.isin([2018])].set_index('Date')
#year2018 = temp[temp.Date.dt.year.isin([2018])].set_index('Date')

temp = temp.reset_index()

temp['weekdays'] = temp['Date'].dt.dayofweek
temp['month'] = temp['Date'].dt.month 
temp['year'] = temp['Date'].dt.year

dmap = {0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}
temp['weekdays'] = temp['weekdays'].map(dmap)

month_dict = {1 :"Jan",2 :"Feb",3 :"Mar",4 :"Apr",5 : "May",6 : "Jun",7 : "Jul",8 :"Aug",9 :"Sep",10 :"Oct",11 :"Nov",12 :"Dec"}
temp['month'] = temp['month'].map(month_dict)

del temp['Date']


# In[12]:


year2013_to_2018.iplot(kind = 'scatter', xTitle='Year 2013 to Year 2018',  yTitle = "# of people killed, injured and affected", title ="Year 2013 to year 2018 Gun Violence incidents in USA")


# In[13]:


temp1 = temp.groupby('year').sum()
temp1 = temp1.reset_index()
temp1[['year','N_killed','N_injured', 'Total_Number_Of_Affected_People']].set_index('year').iplot(kind = 'bar', xTitle = 'Year', yTitle = "# of people killed, injured and affected", title ="Gun Violence Incidents in USA - By Year")


# In[14]:


temp1 = temp.groupby('month').sum()
temp1 = temp1.reset_index()
temp1[['month','N_killed','N_injured', 'Total_Number_Of_Affected_People']].set_index('month').iplot(kind = 'bar', xTitle = 'Month', yTitle = "# of people killed, injured and affected", title ="Gun Violence Incidents in USA - By Month")


# In[15]:


temp1 = temp.groupby('weekdays').sum()
temp1 = temp1.reset_index()
temp1[['weekdays','N_killed','N_injured', 'Total_Number_Of_Affected_People']].set_index('weekdays').iplot(kind = 'bar', xTitle = 'Day Of Week', yTitle = "# of people killed, injured and affected", title ="Gun Violence Incidents in USA - By Weekday")


# Analysing Gun Violence Trends based on location

# In[16]:


df.groupby(df.State).size().nlargest(10).plot(kind='barh', figsize=(8, 6),
    title='Top 10 states with the highest number of gun incidents')


# In[17]:


df.groupby(df.State).size().nsmallest(10).plot(kind='barh', figsize=(8, 6),
    title='Top 10 states with the lowest number of gun incidents')


# In[18]:


df.groupby(df.City_or_county).size().nlargest(10).plot(kind='barh', figsize=(8, 6),
    title='Top 10 cities with the highest number of gun incidents')


# In[19]:


data1 = df[['State', 'N_killed']].groupby(['State'], 
                                   as_index=False).sum().sort_values(by='N_killed', ascending=False).head(20)
data2 = df[['State', 'N_injured']].groupby(['State'], 
                                   as_index=False).sum().sort_values(by='N_injured', ascending=False).head(20)
trace1 = go.Bar(
    x=data1.State,
    y=data1.N_killed,
    name = 'State name with # of people killed'
)
trace2 = go.Bar(
    x=data2.State,
    y=data2.N_injured,
    name = 'State name with # of people injured'
)

fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Top States where maximum people killed', 'Top States where maximum people injured'))
                                                          

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
                          
fig['layout']['xaxis1'].update(title='State Name')
fig['layout']['xaxis2'].update(title='State Name')

fig['layout']['yaxis1'].update(title='# of people killed')
fig['layout']['yaxis2'].update(title='# of people injured')
                          
fig['layout'].update(height=500, width=1100, title='Top States where maximum people killed and injured')
iplot(fig, filename='simple-subplot')


# In[20]:


statdf = df.reset_index().groupby(by=['State']).agg({'loss':'sum', 'year':'count'}).rename(columns={'year':'count'})
statdf['state'] = statdf.index

trace1 = go.Bar(
    x=statdf['state'],
    y=statdf['count'],
    name='Count of Incidents',
    marker=dict(color='rgb(255,10,225)'),
    opacity=0.6
)
trace2 = go.Bar(
    x=statdf['state'],
    y=statdf['loss'],
    name='Total Loss',
    marker=dict(color='rgb(58,22,225)'),
    opacity=0.6
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group',
    margin=dict(b=150),
    legend=dict(dict(x=-.1, y=1.2)),
    title = 'State wise number of Gun Violence Incidents and Total Loss',
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='grouped-bar')


# EXPLORATION OF KEY INCIDENT CHARACTERISTICS

# In[21]:


from collections import Counter
key_text = "||".join(df['Incident_characteristics'].dropna()).split("||")
incidents = Counter(key_text).most_common(20)
x1 = [x[0] for x in incidents]
y1 = [x[1] for x in incidents]

trace1 = go.Bar(
    x=y1[::-1],
    y=x1[::-1],
    name='Incident Characterisitcs',
    marker=dict(color='purple'),
    opacity=0.3,
    orientation="h"
)
data = [trace1]
layout = go.Layout(
    barmode='group',
    margin=dict(l=350),
    width=800,
    height=600,
    legend=dict(dict(x=-.1, y=1.2)),
    title = 'Key Incident Characteristics',
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='grouped-bar')


# In[22]:


temp = df["Location_description"].value_counts().head(20)
temp.iplot(kind='bar', xTitle = 'Place name', yTitle = "# of incidents", title = 'Top Places in the cities with highest number of Gun Violence')


# In[23]:


temp1 = df['N_guns_involved'].dropna().apply(lambda x : "4+" if x>4 else str(x))
temp = temp1.value_counts()
df1 = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df1.iplot(kind='pie',labels='labels',values='values', title='Number of Guns Involved During Violence')


# In[24]:


parse_list = lambda x: [y.split('::')[-1] for y in x.split('||')]


# In[25]:


def count_subjects(row, column, target):
    status_lst = parse_list(row[column])
    type_lst = parse_list(row['Participant_type'])
    return sum(_type == 'Subject-Suspect' and _status == target for _status, _type in zip(status_lst, type_lst))


# In[26]:


subject_gender = {target: df.dropna(subset=['Participant_type', 'Participant_gender']).apply(lambda x: count_subjects(x, 'Participant_gender', target), axis=1).sum() for target in ['Male', 'Female']}
fig, ax = plt.subplots(1, 1, figsize=(8, 6)) 
ax.pie(subject_gender.values(), labels=subject_gender.keys())
ax.axis('equal')
ax.set_title('Suspect subject gender breakout')


# In[27]:


subject_status = {target: df.dropna(subset=['Participant_status', 'Participant_type']).apply(lambda x: count_subjects(x, 'Participant_status', target), axis=1).sum() for target in ['Arrested', 'Killed', 'Unharmed']}
fig, ax = plt.subplots(1, 1, figsize=(8, 6)) 
ax.pie(subject_status.values(), labels=subject_status.keys())
ax.axis('equal')
ax.set_title('Suspect subject type breakout')


# In[30]:


df_18=df.ix[df['year']==2018]


# In[31]:


df_ts=df_18[['N_killed','Date']]


# In[32]:


df_ts.index=df_18['date']


# In[35]:


from fbprophet import Prophet
sns.set(font_scale=1) 
df_date_index = df_18[['Date','N_killed']]
df_date_index = df_date_index.set_index('Date')
df_prophet = df_date_index.copy()
df_prophet.reset_index(drop=False,inplace=True)
df_prophet.columns = ['ds','y']

m = Prophet()
m.fit(df_prophet)
future = m.make_future_dataframe(periods=270,freq='D')
forecast = m.predict(future)
fig = m.plot(forecast)


# In[36]:


m.plot_components(forecast);


# GUN LAWS IMPACT ON GUN VIOLENCE

# Let us analyze how strict are the gun laws in different states, and how does they correlate with the number of gun violence incidents in those states. 
# We have taken the gun laws data for the following years: 2014, 2015, 2016, 2017, 2018 from the following source:
# 
# Source: https://statefirearmlaws.org/national-data/
# 
# Rise of Gun Violence Laws in different States : 2014 - 2018 

# In[27]:


state_to_code = {'District of Columbia' : 'dc','Mississippi': 'MS', 'Oklahoma': 'OK', 'Delaware': 'DE', 'Minnesota': 'MN', 'Illinois': 'IL', 'Arkansas': 'AR', 'New Mexico': 'NM', 'Indiana': 'IN', 'Maryland': 'MD', 'Louisiana': 'LA', 'Idaho': 'ID', 'Wyoming': 'WY', 'Tennessee': 'TN', 'Arizona': 'AZ', 'Iowa': 'IA', 'Michigan': 'MI', 'Kansas': 'KS', 'Utah': 'UT', 'Virginia': 'VA', 'Oregon': 'OR', 'Connecticut': 'CT', 'Montana': 'MT', 'California': 'CA', 'Massachusetts': 'MA', 'West Virginia': 'WV', 'South Carolina': 'SC', 'New Hampshire': 'NH', 'Wisconsin': 'WI', 'Vermont': 'VT', 'Georgia': 'GA', 'North Dakota': 'ND', 'Pennsylvania': 'PA', 'Florida': 'FL', 'Alaska': 'AK', 'Kentucky': 'KY', 'Hawaii': 'HI', 'Nebraska': 'NE', 'Missouri': 'MO', 'Ohio': 'OH', 'Alabama': 'AL', 'Rhode Island': 'RI', 'South Dakota': 'SD', 'Colorado': 'CO', 'New Jersey': 'NJ', 'Washington': 'WA', 'North Carolina': 'NC', 'New York': 'NY', 'Texas': 'TX', 'Nevada': 'NV', 'Maine': 'ME'}
laws_2014 = {'Mississippi': 5, 'Oklahoma': 9, 'Delaware': 38, 'Minnesota': 39, 'Illinois': 65, 'Arkansas': 11, 'New Mexico': 10, 'Ohio': 15, 'Indiana': 11, 'Maryland': 63, 'Louisiana': 12, 'Texas': 20, 'Wyoming': 6, 'Tennessee': 24, 'Arizona': 8, 'Wisconsin': 24, 'Michigan': 20, 'Kansas': 8, 'Utah': 11, 'Virginia': 12, 'Oregon': 24, 'Connecticut': 85, 'Montana': 4, 'California': 100, 'Idaho': 5, 'West Virginia': 24, 'South Carolina': 12, 'New Hampshire': 10, 'Massachusetts': 101, 'Vermont': 3, 'Georgia': 6, 'North Dakota': 14, 'Pennsylvania': 37, 'Florida': 21, 'Alaska': 3, 'Kentucky': 7, 'Hawaii': 78, 'Nebraska': 22, 'Missouri': 8, 'Iowa': 25, 'Alabama': 10, 'Rhode Island': 43, 'South Dakota': 5, 'Colorado': 30, 'New Jersey': 67, 'Washington': 41, 'North Carolina': 30, 'New York': 76, 'Nevada': 11, 'Maine': 11}
laws_2015 = {'Mississippi': 2, 'Oklahoma': 9, 'Delaware': 38, 'Minnesota': 41, 'Illinois': 65, 'Arkansas': 11, 'New Mexico': 10, 'Ohio': 16, 'Indiana': 11, 'Maryland': 64, 'Louisiana': 12, 'Texas': 20, 'Wyoming': 6, 'Tennessee': 24, 'Arizona': 8, 'Wisconsin': 23, 'Michigan': 21, 'Kansas': 4, 'Utah': 11, 'Virginia': 12, 'Oregon': 33, 'Connecticut': 85, 'Montana': 4, 'California': 102, 'Idaho': 6, 'West Virginia': 24, 'South Carolina': 12, 'New Hampshire': 10, 'Massachusetts': 101, 'Vermont': 6, 'Georgia': 6, 'North Dakota': 14, 'Pennsylvania': 37, 'Florida': 21, 'Alaska': 3, 'Kentucky': 7, 'Hawaii': 78, 'Nebraska': 22, 'Missouri': 8, 'Iowa': 25, 'Alabama': 10, 'Rhode Island': 43, 'South Dakota': 5, 'Colorado': 30, 'New Jersey': 67, 'Washington': 41, 'North Carolina': 30, 'New York': 76, 'Nevada': 15, 'Maine': 11}
laws_2016 = {'Mississippi': 2, 'Oklahoma': 9, 'Delaware': 39, 'Minnesota': 41, 'Illinois': 65, 'Arkansas': 11, 'New Mexico': 10, 'Ohio': 16, 'Indiana': 12, 'Maryland': 64, 'Louisiana': 12, 'Texas': 18, 'Wyoming': 6, 'Tennessee': 22, 'Arizona': 8, 'Wisconsin': 23, 'Michigan': 21, 'Kansas': 4, 'Utah': 11, 'Virginia': 13, 'Oregon': 35, 'Connecticut': 90, 'Montana': 4, 'California': 104, 'Idaho': 2, 'West Virginia': 18, 'South Carolina': 12, 'New Hampshire': 10, 'Massachusetts': 101, 'Vermont': 6, 'Georgia': 6, 'North Dakota': 14, 'Pennsylvania': 37, 'Florida': 21, 'Alaska': 3, 'Kentucky': 7, 'Hawaii': 79, 'Nebraska': 22, 'Missouri': 7, 'Iowa': 25, 'Alabama': 10, 'Rhode Island': 43, 'South Dakota': 5, 'Colorado': 30, 'New Jersey': 67, 'Washington': 43, 'North Carolina': 30, 'New York': 76, 'Nevada': 15, 'Maine': 11}
laws_2017 = {'Mississippi': 2, 'Oklahoma': 9, 'Delaware': 40, 'Minnesota': 41, 'Illinois': 65, 'Arkansas': 11, 'New Mexico': 10, 'Ohio': 16, 'Indiana': 12, 'Maryland': 64, 'Louisiana': 13, 'Texas': 18, 'Wyoming': 6, 'Tennessee': 22, 'Arizona': 8, 'Wisconsin': 23, 'Michigan': 21, 'Kansas': 4, 'Utah': 13, 'Virginia': 13, 'Oregon': 35, 'Connecticut': 90, 'Montana': 4, 'California': 106, 'Idaho': 2, 'West Virginia': 18, 'South Carolina': 12, 'New Hampshire': 9, 'Massachusetts': 101, 'Vermont': 6, 'Georgia': 6, 'North Dakota': 10, 'Pennsylvania': 37, 'Florida': 21, 'Alaska': 3, 'Kentucky': 7, 'Hawaii': 79, 'Nebraska': 22, 'Missouri': 2, 'Iowa': 24, 'Alabama': 10, 'Rhode Island': 53, 'South Dakota': 5, 'Colorado': 30, 'New Jersey': 75, 'Washington': 43, 'North Carolina': 30, 'New York': 76, 'Nevada': 21, 'Maine': 11}
laws_2018 = {'Mississippi': 2, 'Oklahoma': 9, 'Delaware': 40, 'Minnesota': 41, 'Illinois': 65, 'Arkansas': 11, 
             'New Mexico': 10, 'Ohio': 16, 'Indiana': 12, 'Maryland': 64, 'Louisiana': 13, 'Texas': 18, 'Wyoming': 6, 
             'Tennessee': 22, 'Arizona': 8, 'Wisconsin': 23, 'Michigan': 21, 'Kansas': 4, 'Utah': 13, 'Virginia': 13, 
             'Oregon': 37, 'Connecticut': 90, 'Montana': 4, 'California': 106, 'Idaho': 2, 'West Virginia': 18, 
             'South Carolina': 12, 'New Hampshire': 9, 'Massachusetts': 101, 'Vermont': 6, 'Georgia': 6, 'North Dakota': 10, 
             'Pennsylvania': 37, 'Florida': 21, 'Alaska': 3, 'Kentucky': 7, 'Hawaii': 79, 'Nebraska': 22, 'Missouri': 2, 
             'Iowa': 24, 'Alabama': 10, 'Rhode Island': 53, 'South Dakota': 5, 'Colorado': 30, 'New Jersey': 75, 
             'Washington': 43, 'North Carolina': 30, 'New York': 76, 'Nevada': 21, 'Maine': 11}

y1, y2, y3, y4, y5 = [], [], [], [], []
x1 = []
for x, y in laws_2014.items():
    y1.append(y)
    y2.append(laws_2015[x])
    y3.append(laws_2016[x])
    y4.append(laws_2017[x])
    y5.append(laws_2018[x])
    x1.append(x)

trace1 = go.Bar(x=x1,y=y1,name='2014', marker=dict(color="#f27da6"))
trace2 = go.Bar(x=x1,y=y2,name='2015', marker=dict(color="#7f9bef"))
trace3 = go.Bar(x=x1,y=y3,name='2016', marker=dict(color="#94e8ba"))
trace4 = go.Bar(x=x1,y=y4,name='2017', marker=dict(color="#e8d54a"))
trace5 = go.Bar(x=x1,y=y5,name='2018', marker=dict(color="#87d57b"))

data = [trace1, trace2, trace3, trace4, trace5]
layout = go.Layout(barmode='group', title="Rise of Gun Violence Laws : 2014 - 2018")
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='grouped-bar')


# In[28]:


states_df = df[df['year'] == 2017]['State'].value_counts()
statesdf = pd.DataFrame()
statesdf['state'] = states_df.index
statesdf['counts'] = states_df.values
statesdf['laws'] = statesdf['state'].apply(lambda x : laws_2017[x] if x in laws_2017 else "")

statesdf['state'] = statesdf['state'].apply(lambda x : state_to_code[x])

data = [
    {
        'x': statesdf['laws'],
        'y': statesdf['counts'],
        'mode': 'markers+text',
        'text' : statesdf['state'],
        'textposition' : 'bottom center',
        'marker': {
            'color': "#7ae6ff",
            'size': 15,
            'opacity': 0.9
        }
    }
]

layout = go.Layout(title="Gun Laws vs Gun Violence Incidents - 2017", 
                   xaxis=dict(title='Total Gun Laws'),
                   yaxis=dict(title='Gun Violence Incidents')
                  )
fig = go.Figure(data = data, layout = layout)
iplot(fig, filename='scatter-colorscale')

