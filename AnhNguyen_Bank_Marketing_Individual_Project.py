
# coding: utf-8

# In[1]:


#Import Dataset and take a look 


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# conda install -c anaconda pydot
# conda install graphviz (in case you have "dot.exe not found in path" error)
import pydot
from IPython.display import Image

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.externals.six import StringIO  
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error,confusion_matrix, classification_report

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-white')

df = pd.read_csv('bank-additional-full.csv', sep = ";")


# In[3]:


#Print out first 5 lines of the dataset 
print(df.head(5))


# In[4]:


# Step 1: explore data

## cat feature: count number of each item
# num feature: histogram 
# outcome feature: count (how many yes, how many no)

# Step 2: remove missing value for used features - based on step 1
## job: remove unknown
## .....


# Step 3: data transformation/reformat


# In[5]:


# explore categorical features
for col_name in df.columns:
    if df[col_name].dtypes == 'object':
        unique_cat = len(df[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} unique categories".format(col_name=col_name, unique_cat=unique_cat))
        print(df[col_name].value_counts())
        print('\n--------------------\n')


# In[6]:


# using histogram to explore distribution of numerical features

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

def plot_histogram(x):
    plt.hist(x, color='blue', alpha=0.5)
    plt.title("Histogram of '{var_name}'".format(var_name=x.name))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()


for col_name in df.columns:
    if df[col_name].dtypes != 'object':
        print('\n--------------------\n')
        plot_histogram(df[col_name])


# In[7]:


# Remove missing values
## all unknown value in the dataset is considered missing-value

selected_features = ['job', 'marital', 'education', 'housing', 'loan', 'age', 'campaign','y']
selected_cat_features = ['job', 'marital', 'education', 'housing', 'loan']

for f in selected_cat_features:
    df = df.loc[ df[f] != 'unknown', ]


#  Mapping and Factorizing 
    
df = df[selected_features]

df.y = df.y.map({'yes':1, 'no':0})

df.loan = df.loan.map({'yes':1, 'no':0})

df.housing = df.housing.map({'yes':1, 'no':0})

df.marital = pd.factorize(df.marital)[0]

df.job = pd.factorize(df.job)[0]

df.education = pd.factorize(df.education)[0]

df.info()

# Split the dataset into testing and Training datasets

y = df.y
X = df.drop(['y'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)


#Fitting Decision Tree

clf = DecisionTreeClassifier(max_depth=6)
clf.fit(X, y)

print(classification_report(y, clf.predict(X)))

# Printing out Decision Tree
graph, = print_tree(clf, features=X.columns, class_names=['No', 'Yes'])
Image(graph.create_png())


clf.fit(X_train, y_train)
pred = clf.predict(X_test)

cm = pd.DataFrame(confusion_matrix(y_test, pred).T, index=['No', 'Yes'], columns=['No', 'Yes'])
cm.index.name = 'Predicted'
cm.columns.name = 'True'
cm

# Confusion Matrix 
print(classification_report(y_test, pred))
