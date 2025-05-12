# Data Exploration and Processing 

# Import libraries 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes = True)
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")

# Importing data
dataset_path = "aerofit_treadmill_data.csv"
aerofit_df = pd.read_csv(dataset_path)

# Reading dataframe
print(aerofit_df.head())

# Shape of the dataframe
aerofit_df.shape

# Name of each column in dataframe
aerofit_df.columns

# Name of each column in dataframe
aerofit_df.columns

aerofit_df['Product'] = aerofit_df['Product'].astype('category')
aerofit_df['Gender'] = aerofit_df['Gender'].astype('category')
aerofit_df['MaritalStatus'] = aerofit_df['MaritalStatus'].astype('category')

aerofit_df.info()

#aerofit_df.skew()

#Statistical Summary 

aerofit_df.describe(include = 'all')

'''
Observations:

There are no missing values in the data.
There are 3 unique products in the dataset.
KP281 is the most frequent product.
Minimum & Maximum age of the person is 18 & 50, mean is 28.79, and 75% of persons have an age less than or equal to 33.
Most of the people are having 16 years of education i.e. 75% of persons are having education <= 16 years.
Out of 180 data points, 104's gender is Male and rest are the Female.
Standard deviation for Income & Miles is very high. These variables might have outliers in them.
'''

# Missing value detection
aerofit_df.isna().sum() # No missing values detected in DataFrame

# Checking duplicate values in the dataset
aerofit_df.duplicated(subset=None,keep='first').sum() # No duplicate values in the dataset

#Non-Graphical Analysis
#Value Counts
aerofit_df["Product"].value_counts()
aerofit_df["Gender"].value_counts()

