#!/usr/bin/env python
# coding: utf-8

# # Acquire and Prep - Wrangle

# In[1]:


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import env
from env import host, user, password

from sklearn.model_selection import train_test_split


# In[2]:


def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


# In[3]:


# function to query database and return zillow df
def get_data_from_sql():
    query = """
    SELECT bedroomcnt as bedrooms, bathroomcnt as bathrooms, calculatedfinishedsquarefeet as square_feet, 
    taxvaluedollarcnt FROM properties_2017
    JOIN predictions_2017 as p USING(parcelid)
    WHERE transactiondate < '2018-01-01' AND propertylandusetypeid LIKE '261'
    """
    df = pd.read_sql(query, get_connection('zillow'))
    return df


# In[4]:


df = get_data_from_sql()


# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


df.describe().T


# In[8]:


df.info()


# In[9]:


df.value_counts()


# # Prepare the Data

# In[10]:


# Find the total number of Null values in each column of our DataFrame.

df.isnull().sum()


# In[11]:


# Find columns with missing values and the total of missing values.

missing = df.isnull().sum()
missing[missing > 0]


# In[12]:


# show all columns with missing values
missing.head()


# In[13]:


# Check for any Null values in each column of our DataFrame.

df.isnull().any()



# In[14]:


# Return the names for any columns in our DataFrame with any Null values.

df.columns[df.isnull().any()]



# Finding Odd Values
# 
# Let's find the odd value that is causing this numeric column to be coerced into an object data type.

# In[15]:


df.bedrooms.value_counts(dropna=False, ascending=True)


# In[16]:


# check value counts for bedrooms column

df['bedrooms'].value_counts(ascending=False)


# In[17]:


#check value counts for bathrooms column

df['bathrooms'].value_counts(ascending=True)



# In[18]:


# Replace a whitespace sequence or empty with a NaN value and reassign this manipulation to df.

df = df.replace(r'^\s*$', np.nan, regex=True)


# In[19]:


# Now .info() shows us that bedrooms has a Null value instead of a whitespace disguised as a non-null value.

df.isnull().any()


# ## Drop Null Values

# In[20]:


df.shape


# In[21]:


df = df.dropna()
df.info()


# In[22]:


# confirmation that above code worked to drop nulls
df.isnull().any()


# In[23]:


df.shape


# In[24]:


# drop duplicates
df = df.drop_duplicates()


# In[25]:


# confirm cell above that duplicates have been dropped
df.shape


# In[26]:


#function to remove outliers in x columns

def remove_outliers(df, k, col_list):
    ''' 
    Takes in a df, k, and list of columns returns
    a df with removed outliers
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df


# In[27]:


# use above function to remove outliers for columns listed and apply to new df
df = remove_outliers(df, 1.5, ['bedrooms', 'bathrooms', 'square_feet', 'taxvaluedollarcnt'])
df


# In[28]:


# confirm outliers removed
df.shape


# In[29]:


# convert dtypes to int

df = df.astype('int')
df.info()


# ## Visualize Distributions
# 
# 

# In[30]:


plt.figure(figsize=(16, 3))

# List of columns
cols = ['bedrooms', 'bathrooms', 'square_feet','taxvaluedollarcnt']

for i, col in enumerate(cols):

    # i starts at 0, but plot nos should start at 1
    plot_number = i + 1 

    # Create subplot.
    plt.subplot(1,4, plot_number)

    # Title with column name.
    plt.title(col)

    # Display histogram for column.
    df[col].hist(bins=5)

    # Hide gridlines.
    plt.grid(False)


# In[31]:


plt.figure(figsize=(10,6))

# Create boxplots for all.
sns.boxplot(data=df)
plt.show()


# In[32]:


def show_boxplot(df):
    ''' Gets boxplots of acquired continuous variables'''
    
    # List of columns
    cols = ['bedrooms', 'bathrooms', 'square_feet', 'taxvaluedollarcnt']

    plt.figure(figsize=(16, 3))

    for i, col in enumerate(cols):

        # i starts at 0, but plot should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display boxplot for column.
        sns.boxplot(data=df[[col]])

        # Hide gridlines.
        plt.grid(False)

        # sets proper spacing between plots
        plt.tight_layout()

    plt.show()


# In[33]:


show_boxplot(df)


# ## Create function for acquire and prep
# 

# In[34]:


#will update

def wrangle_zillow():
    '''
    Read student_grades csv file into a pandas DataFrame,
    drop student_id column, replace whitespaces with NaN values,
    drop any rows with Null values, convert all columns to int64,
    return cleaned student grades DataFrame.
    '''
    # Acquire data from csv file.
    df = get_data_from_sql()
    
    # Replace white space values with NaN values.
    df = df.replace(r'^\s*$', np.nan, regex=True)
    
    # Drop any Duplicates
    df = df.drop_duplicates()

    # Drop all rows with NaN values.
    df = df[df.bathrooms != 0]
    df = df[df.bedrooms != 0]
    
    # Remove Outliers
    df = remove_outliers(df, 1.5, ['bedrooms', 'bathrooms', 'square_feet', 'taxvaluedollarcnt'])

    
    # Convert datatypes to int
    df = df.astype('int')
    

    # clean data with split
    train, test = train_test_split(df, test_size = .2, random_state = 123)
    train, validate = train_test_split(train, test_size = .3, random_state = 123)
    
    return df 


# In[35]:


wrangle_zillow = wrangle_zillow()


# In[36]:


wrangle_zillow.head()


# In[37]:


wrangle_zillow.info()


# In[38]:


wrangle_zillow.describe()


# In[39]:


wrangle_zillow.shape


# In[40]:


## For Splitting:

# Only use train for exploration and for fitting
# Only use validate to validate models after fitting on train
# Only use test to test best model 

# Split data

# 20% test, 80% train_validate
# then of the 80% train_validate: 30% validate, 70% train.

train, test = train_test_split(wrangle_zillow, test_size = .2, random_state = 123)
train, validate = train_test_split(train, test_size = .3, random_state = 123)


# In[41]:


# Validate my split.

print(f'train -> {train.shape}')
print(f'validate -> {validate.shape}')
print(f'test -> {test.shape}')


# In[42]:


def split_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames
    return train, validate, test DataFrames.
    '''
    train, test = train_test_split(wrangle_zillow, test_size = .2, random_state = 222)
    train, validate = train_test_split(train, test_size = .3, random_state = 222)
    
    return train, validate, test


# In[43]:


train, validate, test = split_data(wrangle_zillow)


# In[44]:


print(f'train -> {train.shape}')
print(f'validate -> {validate.shape}')
print(f'test -> {test.shape}')


# In[45]:


train.head()


# In[46]:


train.shape


# In[ ]:





# # Exploration
# 
# 

# - What is the relationship between square feet and taxvaluedollarcount?
#     - Is it a linear relationship or is there no relationship?
#     
# - What is the relationship between bedroom count and taxvaluedollarcount?
# 
# - What is the relationship between bathroom count and taxvaluedollarcount?
# 
# - What is the relationship between bedroom 
# 

# In[ ]:




