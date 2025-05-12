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
aerofit_df["MaritalStatus"].value_counts()


#Unique Attributes
aerofit_df.nunique()
aerofit_df["Product"].unique()
aerofit_df["Age"].unique()
aerofit_df["Gender"].unique()
aerofit_df["Education"].unique()
aerofit_df["MaritalStatus"].unique()
aerofit_df["Usage"].unique()
aerofit_df["Income"].unique()
aerofit_df["Fitness"].unique()
aerofit_df["Miles"].unique()

#Graphical Analysis 

#Univariate Analysis - Numerical Variables

#Distance Plot 
fig, axis = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))
fig.subplots_adjust(top=1.2)

sns.distplot(aerofit_df['Age'], kde=True, ax=axis[0,0])
sns.distplot(aerofit_df['Education'], kde=True, ax=axis[0,1])
sns.distplot(aerofit_df['Usage'], kde=True, ax=axis[1,0])
sns.distplot(aerofit_df['Fitness'], kde=True, ax=axis[1,1])
sns.distplot(aerofit_df['Income'], kde=True, ax=axis[2,0])
sns.distplot(aerofit_df['Miles'], kde=True, ax=axis[2,1])
plt.show()

'''
Observations:

Both Miles and Income have significant outliers based on the above distribution.

Also Miles and Income are "right-skewed distribution" which means the mass of the distribution is concentrated on the left of the figure.

Customer with fitness level 3 buy a major chuck of treadmills.

Majority of Customers fall within the $ 45,000 - $ 60,000 income range
'''

#Count Plot
fig, axis = plt.subplots(nrows=3, ncols=2, figsize=(20, 12))
fig.subplots_adjust(top=1.2)

sns.countplot(data=aerofit_df, x="Age", ax=axis[0,0])
sns.countplot(data=aerofit_df, x="Education", ax=axis[0,1])
sns.countplot(data=aerofit_df, x="Usage", ax=axis[1,0])
sns.countplot(data=aerofit_df, x="Fitness", ax=axis[1,1])
sns.countplot(data=aerofit_df, x="Income", ax=axis[2,0])
sns.countplot(data=aerofit_df, x="Miles", ax=axis[2,1])
plt.show()

'''
Observations:

Young people at age of 25 are more conscious about health and are using treadmills more than old aged people.
'''

#Box Plot
fig, axis = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))
fig.subplots_adjust(top=1.2)

sns.boxplot(data=aerofit_df, x="Age", orient='h', ax=axis[0,0])
sns.boxplot(data=aerofit_df, x="Education", orient='h', ax=axis[0,1])
sns.boxplot(data=aerofit_df, x="Usage", orient='h', ax=axis[1,0])
sns.boxplot(data=aerofit_df, x="Fitness", orient='h', ax=axis[1,1])
sns.boxplot(data=aerofit_df, x="Income", orient='h', ax=axis[2,0])
sns.boxplot(data=aerofit_df, x="Miles", orient='h', ax=axis[2,1])
plt.show()

'''
Observations:

Age, Education, and Usage have very few outliers.
While Income and Miles have more outliers.
'''

#Univariate Analysis - Categorical Variables

#Count Plot
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
sns.countplot(data=aerofit_df, x='Product', ax=axs[0])
sns.countplot(data=aerofit_df, x='Gender', ax=axs[1])
sns.countplot(data=aerofit_df, x='MaritalStatus', ax=axs[2])

axs[0].set_title("Product - counts", pad=10, fontsize=14)
axs[1].set_title("Gender - counts", pad=10, fontsize=14)
axs[2].set_title("MaritalStatus - counts", pad=10, fontsize=14)
plt.show()

'''Observations

KP281 is the most frequent product and best-selling product.
In Gender, there are more Males who are using treadmills than Females.
The treadmills are more likely to be purchased by partnered people
'''

#Bivariate Analysis - Checking if features have any effect on the products purchased 

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))
sns.countplot(data=aerofit_df, x='Product', hue='Gender', edgecolor="0.15", ax=axs[0])
sns.countplot(data=aerofit_df, x='Product', hue='MaritalStatus', edgecolor="0.15", ax=axs[1])
sns.countplot(data=aerofit_df, x='Age', hue='Product', edgecolor="0.15", ax=axs[2])
axs[0].set_title("Product vs Gender", pad=10, fontsize=14)
axs[1].set_title("Product vs MaritalStatus", pad=10, fontsize=14)
plt.show()

'''
Obervations

Product vs Gender
Equal number of males and females have purchased KP281 product and Almost same for the product KP481
Most of the Male customers have purchased the KP781 product.
Product vs MaritalStatus

Customer who is Partnered, is more likely to purchase the product.
Age vs Product

Customers with age of 25 are more likely to purchase the KP481 product.
'''

attributes = ['Age', 'Education', 'Usage', 'Fitness', 'Income', 'Miles']
sns.set(color_codes = True)
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
fig.subplots_adjust(top=1.2)
count = 0
for i in range(2):
    for j in range(3):
        sns.boxplot(data=aerofit_df, x='Product', y=attributes[count], ax=axs[i,j])
        axs[i,j].set_title(f"Product vs {attributes[count]}", pad=12, fontsize=13)
        count += 1

'''
Observations:

Product vs Age
Customers purchasing products KP281 & KP481 are having same Age median value.
Customers whose age lies between 25-30, are more likely to buy the KP781 product
Product vs Education
Customers whose Education is greater than 16, have more chances to purchase the KP781 product.
While the customers with Education less than 16 have equal chances of purchasing KP281 or KP481.
Product vs Usage
Customers who are planning to use the treadmill greater than 4 times a week, are more likely to purchase the KP781 product.
While the other customers are likely to purchase KP281 or KP481.
Product vs Fitness
The more the customer is fit (fitness >= 3), the higher the chances of the customer purchasing the KP781 product.
Product vs Income
The higher the Income of the customer (Income >= 60000), the higher the chances of the customer purchasing the KP781 product.
Product vs Miles
If the customer expects to walk/run greater than 120 Miles per week, it is more likely that the customer will buy the KP781 product.
'''

#Multivariate Analysis

attributes = ['Age', 'Education', 'Usage', 'Fitness', 'Income', 'Miles']
sns.set(color_codes = True)
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(18, 12))
fig.subplots_adjust(top=1.3)
count = 0
for i in range(3):
    for j in range(2):
        sns.boxplot(data=aerofit_df, x='Gender', y=attributes[count], hue='Product', ax=axs[i,j])
        axs[i,j].set_title(f"Product vs {attributes[count]}", pad=12, fontsize=13)
        count += 1


'''Obervations

Females planning to use the treadmill 3-4 times a week, are more likely to buy the KP481 product
'''

#Correlation Analysis

#aerofit_df.cov()
#aerofit_df.corr()
'''
#Heatmaps
fig, ax = plt.subplots(figsize = (10,10))
sns.set(color_codes = True)
sns.heatmap(aerofit_df.corr(), ax=ax, annot = True, linewidths = 0.05, fmt ='0.2f')
plt.show()
'''

'''
Observations:

(Miles & Fitness) and (Miles & Usage) attributes are highly correlated, which means if a customer's fitness level is high they use more treadmills.

Income and Education shows a strong correlation. High-income and highly educated people prefer the KP781 treadmill which is having advanced features.

There is no correlation between (Usage & Age) or (Fitness & Age) attributes, which mean Age should not be a barrier to using treadmills or specific model of treadmills.
'''

#Pair Plots
sns.pairplot(aerofit_df, hue = "Product")
plt.show()

#Marginal & Conditional Probabilities

aerofit_df1 = aerofit_df[['Product', 'Gender', 'MaritalStatus']].melt()
(aerofit_df1.groupby(['variable', 'value'])[['value']].count() / len(aerofit_df)).mul(100).round(3).astype(str) + '%'

'''
Obervations

Product
44.44% of the customers have purchased KP281 product.
33.33% of the customers have purchased KP481 product.
22.22% of the customers have purchased KP781 product.
Gender
57.78% of the customers are Male.
MaritalStatus
59.44% of the customers are Partnered.
'''

#What is the probability of a customer based on Gender ( Male or Female ) buying a certain treadmill Product?

def p_prod_given_gender(gender, print_marginal=False):
    if gender is not "Female" and gender is not "Male":
        return "Invalid gender value."
    
    aerofit_df1 = pd.crosstab(index=aerofit_df['Gender'], columns=[aerofit_df['Product']])
    p_281 = aerofit_df1['KP281'][gender] / aerofit_df1.loc[gender].sum()
    p_481 = aerofit_df1['KP481'][gender] / aerofit_df1.loc[gender].sum()
    p_781 = aerofit_df1['KP781'][gender] / aerofit_df1.loc[gender].sum()
    
    if print_marginal:
        print(f"P(Male): {aerofit_df1.loc['Male'].sum()/len(aerofit_df):.2f}")
        print(f"P(Female): {aerofit_df1.loc['Female'].sum()/len(aerofit_df):.2f}\n")
    
    print(f"P(KP281/{gender}): {p_281:.2f}") 
    print(f"P(KP481/{gender}): {p_481:.2f}")
    print(f"P(KP781/{gender}): {p_781:.2f}\n")
    
p_prod_given_gender('Male', True)
p_prod_given_gender('Female')

#What is the probability of a customer based on MaritalStatus ( Single or Partnered ) buying a certain treadmill Product?

def p_prod_given_mstatus(status, print_marginal=False):
    if status is not "Single" and status is not "Partnered":
        return "Invalid marital status value."
    
    aerofit_df1 = pd.crosstab(index=aerofit_df['MaritalStatus'], columns=[aerofit_df['Product']])
    p_281 = aerofit_df1['KP281'][status] / aerofit_df1.loc[status].sum()
    p_481 = aerofit_df1['KP481'][status] / aerofit_df1.loc[status].sum()
    p_781 = aerofit_df1['KP781'][status] / aerofit_df1.loc[status].sum()
    
    if print_marginal:
        print(f"P(Single): {aerofit_df1.loc['Single'].sum()/len(aerofit_df):.2f}")
        print(f"P(Partnered): {aerofit_df1.loc['Partnered'].sum()/len(aerofit_df):.2f}\n")
    
    print(f"P(KP281/{status}): {p_281:.2f}") 
    print(f"P(KP481/{status}): {p_481:.2f}")
    print(f"P(KP781/{status}): {p_781:.2f}\n")
    
p_prod_given_mstatus('Single', True)
p_prod_given_mstatus('Partnered')

#Product - Gender

product_gender = pd.crosstab(index=aerofit_df['Product'], columns=[aerofit_df['Gender']],margins=True)
product_gender

# Percentage of a Male customer purchasing a treadmill
prob = round((product_gender['Male']['All'] / product_gender['All']['All']),2)
pct = round(prob*100,2)
pct

# Percentage of a Female customer purchasing KP781 treadmill
prob = round((product_gender['Female']['KP781'] / product_gender['All']['All']),2)
pct = round(prob*100,2)
pct

# Probability of a customer being a Female given that Product is KP281
#P(A|B) = P(A,B)/P(B) - Bayes' Theorem
#P(Female|KP281) = P(Female,KP281)/P(KP281)

prob = round((product_gender['Female']['KP281'] / product_gender['All']['KP281']),2)
pct = round(prob*100,2)
pct

'''
Observations:

Female customer prefer to buy KP281 & KP481
50% of female tend to purchase treadmill model KP281
'''

#Product - Age
aerofit_df2 = aerofit_df.copy()
# Extracting 2 new features from Age:
# "AgeCategory" - Teens, 20s, 30s and Above 40s
# "AgeGroup" - 14-20 , 20-30, 30-40 & 40-60

bins = [14,20,30,40,60]
labels =["Teens","20s","30s","Above 40s"]
aerofit_df2['AgeGroup'] = pd.cut(aerofit_df2['Age'], bins)
aerofit_df2['AgeCategory'] = pd.cut(aerofit_df2['Age'], bins,labels=labels)

aerofit_df2.tail()

product_age = pd.crosstab(index=aerofit_df2['Product'], columns=[aerofit_df2['AgeCategory']],margins=True)
product_age

# Percentage of customers with Age between 20s and 30s among all customers
prob = round((product_age['20s']['All'] / product_age['All']['All']),2)
pct = round(prob*100,2)
pct

#Product - Income
aerofit_df3 = aerofit_df.copy()

# Extracting 1 new categorial feature based on the Income:
# "IncomeCategory" - Low Income, Lower-middle Income, Upper-Middle Income and High Income

bins_income = [29000, 35000, 60000, 85000, 105000]
labels_income = ['Low Income','Lower-middle Income','Upper-Middle Income', 'High Income']
aerofit_df3['IncomeCategory'] = pd.cut(aerofit_df3['Income'],bins_income,labels = labels_income)

aerofit_df3.head()

product_income = pd.crosstab(index=aerofit_df3['Product'], columns=[aerofit_df3['IncomeCategory']],margins=True)
product_income

# Percentage of a low-income customer purchasing a treadmill
prob = round(product_income['Low Income']['All'] / product_income['All']['All'],2)
pct = round(prob*100,2)
pct

# Percentage of a high-income customer purchasing KP781 treadmill
prob = round(product_income['High Income']['KP781']/ product_income['All']['All'],2)
pct = round(prob*100,2)
pct

# Percentage of customer with high-income salary buying treadmill given that Product is KP781
prob = round(product_income['High Income']['KP781'] / product_income['All']['KP781'],2)
pct = round(prob*100,2)
pct

#Product - Fitness
product_fitness = pd.crosstab(index=aerofit_df['Product'], columns=[aerofit_df['Fitness']],margins=True)
product_fitness

# Percentage of a customers having fitness level 5 
prob = round((product_fitness[5]['All'] / product_fitness['All']['All']),2)
pct = round(prob*100,2)
pct

# Percentage of a customer with Fitness Level 5 purchasing KP781 treadmill
prob = round((product_fitness[5]['KP781']/ product_fitness['All']['All']),2)
pct = round(prob*100,2)
pct

# Percentage of customer with fitness level 5 buying KP781 treadmill given that Product is KP781
prob = round((product_fitness[5]['KP781']/ product_fitness['All']['KP781']),2)
pct = round(prob*100,2)
pct # customers with fitness level 5 make up 72% of KP781 buyers

#Product - Marital Status
product_marital = pd.crosstab(index=aerofit_df['Product'], columns=[aerofit_df['MaritalStatus']],margins=True)
product_marital

# Percentage of a customers who are partnered using treadmills 
prob = round((product_marital['Partnered']['All'] / product_marital['All']['All']),2)
pct = round(prob*100,2)
pct 

#Outlier Detection


fig, axis = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))
fig.subplots_adjust(top=1.2)

sns.boxplot(data=aerofit_df, x="Age", orient='h', ax=axis[0,0])
sns.boxplot(data=aerofit_df, x="Education", orient='h', ax=axis[0,1])
sns.boxplot(data=aerofit_df, x="Usage", orient='h', ax=axis[1,0])
sns.boxplot(data=aerofit_df, x="Fitness", orient='h', ax=axis[1,1])
sns.boxplot(data=aerofit_df, x="Income", orient='h', ax=axis[2,0])
sns.boxplot(data=aerofit_df, x="Miles", orient='h', ax=axis[2,1])
plt.show()

'''
Obervation

Age, Education and Usage are having very few outliers.
Income and Miles are having more outliers.
Only a few of our customers run more than 180 miles per week
'''

#Outlier Handling for Income Feature

aerofit_df1 = aerofit_df.copy()
#Outlier Treatment: Removing top 5% & bottom 1% of the Column Outlier values
Q3 = aerofit_df1['Income'].quantile(0.75)
Q1 = aerofit_df1['Income'].quantile(0.25)
IQR = Q3-Q1
aerofit_df1 = aerofit_df1[(aerofit_df1['Income'] > Q1 - 1.5*IQR) & (aerofit_df1['Income'] < Q3 + 1.5*IQR)]

sns.boxplot(data=aerofit_df1, x="Income", orient='h')
plt.show()

#Outlier Handling for Miles Feature
#Outlier Treatment: Removing top 5% & bottom 1% of the Column Outlier values
Q3 = aerofit_df1['Miles'].quantile(0.75)
Q1 = aerofit_df1['Miles'].quantile(0.25)
IQR = Q3-Q1
aerofit_df1 = aerofit_df1[(aerofit_df1['Miles'] > Q1 - 1.5*IQR) & (aerofit_df1['Miles'] < Q3 + 1.5*IQR)]

sns.boxplot(data=aerofit_df1, x="Miles", orient='h')
plt.show()

# Before removal of Outliers
aerofit_df.shape

# After removal of Outliers
aerofit_df1.shape

'''
While there are outliers, they may provide many insights for high-end models that can benefit companies more. Therefore, they should not be removed for further analysis.

Actionable Insights & Recommendations
Actionable Insights:
Model KP281 is the best-selling product. 44.0% of all treadmill sales go to model KP281.
The majority of treadmill customers fall within the $ 45,000 - $ 80,000 income bracket.
83% of treadmills are bought by individuals with incomes between $ 35,000 and $ 85,000
There are only 8% of customers with incomes below $ 35000 who buy treadmills.
88% of treadmills are purchased by customers aged 20 to 40.
Miles and Fitness & Miles and Usage are highly correlated, which means if a customer's fitness level is high they use more treadmills.

KP781 is the only model purchased by a customer who has more than 20 years of education and an income of over $ 85,000.

With Fitness level 4 and 5, the customers tend to use high-end treadmills and the average number of miles is above 150 per week
Recommendations:
KP281 & KP481 are popular with customer income of $ 45,000 - $ 60,000 and can be offered by these companies as affordable models.
KP781 should be marketed as a Premium Model and marketing it to high income groups and educational over 20 years market segments could result in more sales.
The KP781 is a premium model, so it is ideally suited for sporty people who have a high average weekly mileage and can be afforded by the high income customers.
Aerofit should conduct market research to determine if it can attract customers with income under $ 35,000 to expand its customer base.
'''