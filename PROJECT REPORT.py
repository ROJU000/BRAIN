# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame as df
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn import preprocessing
from sklearn import utils
from sklearn.preprocessing import LabelEncoder
import matplotlib.ticker as ticker
import seaborn as sns
import statistics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# %%
#import csv file
EC = pd.read_csv("employee_combine.csv")


# %%
#data obersvations
EC.head(20)


# %%
EC.shape


# %%
EC.dept.values


# %%
EC.dtypes


# %%
EC.isna().sum()


# %%
EC.isna().values.any()


# %%
EC_dept = EC.groupby('dept', axis=0).sum()
EC_dept


# %%
EC.describe()


# %%
EC["Attrition_Status"].value_counts()


# %%
EC.isnull().values.any()


# %%
#PREDICTIONS AND ACCURACY DONE BY DECCISIONTREEREGRESSOR, LOGICALREGRESSOR AND METRIC LOSS


# %%
#to analyse the data we have to convert to numerical values so the system can understand
#converting strings to numerical values
##creating labelEncoder
le = preprocessing.LabelEncoder
# Converting string labels into numbers.
EC.dept = LabelEncoder().fit_transform(EC.dept)
EC.salary = LabelEncoder().fit_transform(EC.salary)
EC.Attrition_Status = LabelEncoder().fit_transform(EC.Attrition_Status)


# %%
EC.dtypes


# %%
EC.head(20)


# %%
#attrition by salary
EC_salary = EC.groupby('salary', axis=0).sum()
EC_salary.head()
# 0 = low, 1 = medium, 2 = high


# %%
EC.salary.value_counts()


# %%
#visualization of attrition by salary
plt.figure(figsize=(10,7))
sns.set(style='whitegrid')
sns.countplot(EC["salary"])
plt.title('Attrition of salaries')
plt.xlabel('Salary')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig("salary.png")


# %%
# attrition by promotion in last 5 years
EC['promotion_last_5years'] = EC.promotion_last_5years.replace(to_replace = 0,value = "No promotion") 
EC['promotion_last_5years'] = EC.promotion_last_5years.replace(to_replace = 1,value = "promotion") 
EC_promotion = EC.groupby('promotion_last_5years', axis=0).sum()
EC_promotion.head()


# %%
#visualization of attrition by Promotion in last 5 years
plt.figure(figsize=(10,7))
plt.title("Attrition of Promotion of Employees")
plt.xlabel("Promotion")
plt.ylabel("Count")
sns.set(style="whitegrid")
sns.countplot(EC["promotion_last_5years"])
plt.tight_layout()
plt.savefig("promotion.png")


# %%
# attrition by Work accident
EC['Work_accident'] = EC.Work_accident.replace(to_replace = 0,value = "no_accident") 
EC['Work_accident'] = EC.Work_accident.replace(to_replace = 1,value = "accident") 
EC_work_accident = EC.groupby('Work_accident', axis=0).sum()
EC_work_accident.head()


# %%
#visualization of attrition by Work accident
plt.figure(figsize=(10,7))
plt.title("Attrition by Work Accident")
plt.xlabel("Work_accident")
plt.ylabel("Count")
sns.set(style="whitegrid")
sns.countplot(EC["Work_accident"])
plt.tight_layout()
plt.savefig("work accident.png")


# %%
#attrition by time spent in company
EC_time_spent = EC.groupby('time_spend_company', axis=0).sum()
EC_time_spent


# %%
#visualization of attrition by time spent
sns.set(style="whitegrid")
EC_time_spent['Attrition_Status'].plot(kind='pie',
                            figsize=(10, 17),
                            autopct='%1.1f%%', 
                            startangle=90,
                            explode=(0, 0, 0, 0, 0,0,0,0),
                           shadow=True,             
                            )
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.tight_layout()
plt.savefig("time spent.png")


# %%
#attrition by number of projects
EC_number_project = EC.groupby('number_project', axis=0).sum()
EC_number_project


# %%
sns.set(style="whitegrid")
EC_number_project['Attrition_Status'].plot(kind='pie',
                            figsize=(10, 17),
                            autopct='%1.1f%%', 
                            startangle=90,
                            explode=(0, 0, 0, 0, 0,0),
                           shadow=True,             
                            )
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.tight_layout()
plt.savefig("number of project.png")


# %%
#attirition by department
EC_dept = EC.groupby('dept', axis=0).sum()
EC_dept


# %%
plt.figure(figsize=(10,7))
sns.set(style='darkgrid')
sns.countplot(EC['dept'])
plt.title('Departments of Existing Employee ')
plt.tight_layout()
plt.savefig("dept.png")


# %%
#attrition by statisfactory level for Non-existing employee
EC_stat = EC[11429:]
El_stat = EC_stat[['satisfaction_level']]
El_stat['satisfaction_level'] = pd.cut(El_stat['satisfaction_level'], 
       3, labels=["small", "medium", "high"])
El_stat.satisfaction_level.value_counts()


# %%
#visulisation for statisfactory level of non - existing employees
plt.figure(figsize=(10,7))
sns.set(style='darkgrid')
plt.hist(El_stat['satisfaction_level'])
plt.title('Distribution of satisfaction_level of Employees who left')
plt.xlabel('satisfaction_level')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig("stat non.png")


# %%
# attirition by statisfactory level for existing employee
EC_stat = EC[:]
Ee_stat = EC_stat[['satisfaction_level']]
Ee_stat['satisfaction_level'] = pd.cut(Ee_stat['satisfaction_level'], 
       3, labels=["small", "medium", "high"])
Ee_stat.satisfaction_level.value_counts()


# %%
#visulisation for statisfactory level of existing employees
plt.figure(figsize=(10,7))
sns.set(style='darkgrid')
plt.hist(Ee_stat['satisfaction_level'])
plt.title('Distribution of satisfaction_level of Existing ')
plt.xlabel('satisfaction_level')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig("stat exi.png")


# %%
#hence medium statisfactory level are prone to leave 
#no attrition


# %%
#PREDICTIONS


# %%
# BUILDING MODELS


# %%
EC


# %%
# Converting string labels into numbers.
EC.promotion_last_5years = LabelEncoder().fit_transform(EC.promotion_last_5years)
EC.salary = LabelEncoder().fit_transform(EC.salary)
EC.Work_accident = LabelEncoder().fit_transform(EC.Work_accident)

EC.head()


# %%
# shuffle data set
df1 = EC.sample(frac = 1)
df1.head()


# %%
#drop column ID
df2 = df1.drop('Emp ID',axis=1)


# %%
df2.shape


# %%
x = df2.iloc[:,0:9]
y = df2.iloc[:,9]


# %%
#splitting data set into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=4)


# %%
x_train.shape, y_train.shape, x_test.shape, y_test.shape


# %%
df3 = x_train


# %%
df4 = DecisionTreeRegressor(random_state = 1)
df5 = df4.fit(x_train, y_train)
df5


# %%
#prediction
df6 = df4.predict(x_test)
df6


# %%
#accuracy 1
df6 = df4.score(x_test, y_test)
df6


# %%
#using LogisticRegression
df7 = LogisticRegression(C = 0.01, solver ='liblinear').fit(x_train,y_train)
df7


# %%
#prediction
df8 = df7.predict(x_test)
df8


# %%
#accuracy
df9 = df7.score(x_test, y_test)
df9


# %%
df10  = df7.predict_proba(x_test)
df10


# %%
#metric Log-loss  


# %%
from sklearn.metrics import log_loss
log_loss(y_test, df10)


# %%
##conclusion
#It can be seen that Employees with medium salaries are prone to leave next.
#It can be seen that Employees within 3,4 and 5 years are prone to leave next.
#90% of employees who had no promotion in the last 5 years have left.
#The Employees with more  work accidents had left.
#Most of the Employees who had left the firm were given less no. of projects.
#Most of the Employees in the sales and technical department are more prone to leave the firm.
#Medium satisfaction level Employees are more prone to leave. As it can be clearly seen more no. of present employees have a higher satisfaction level.


# %%
#recommendations


# %%


