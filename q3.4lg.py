import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv("fau_airlines_performance.csv", sep=',')

# check for any missing values
missingdata = df.isnull().values.any()
print("This is missingdata check ", missingdata)
print("********************************")
print("This is df.columns ", df.columns)
print("********************************")


# General plot for the number of employees and performance
sns.distplot(df['PerformanceRating'], hist=True, kde=False,
            bins=int(180/5), color='green',
            hist_kws={'edgecolor': 'black'})
plt.title('Histogram of employees performance')
plt.xlabel('level of performance')
plt.ylabel('number of employees')
plt.show()  

# Assuming 'EmpJobRole' contains the job roles
job_roles = df['EmpJobRole'].unique()

# Calculate the mean performance for each job role and  Plotting a column chart
mean_performance_by_role = df.groupby('EmpJobRole')['PerformanceRating'].mean()
print("Mean Performance for Each Job Role:", mean_performance_by_role)
plt.figure(figsize=(12, 6))
ax = mean_performance_by_role.plot(kind='bar', color=['red', 'black'])
plt.title('Mean Performance by Job Role')
plt.xlabel('Job Role')
plt.ylabel('Mean Performance Rating')
plt.xticks(range(len(job_roles)), job_roles, rotation=45, ha='right')
for i, value in enumerate(mean_performance_by_role):
    ax.text(i, value + 0.01, f'{value:.6f}', ha='center', va='bottom')
plt.show()

# Attributes should be dropped
columns_to_drop = ['Gender', 'EmpNumber']
for col in columns_to_drop:
    if col in df.columns:
        df = df.drop([col], axis=1)
        print(f"The '{col}' column has been dropped.")
    else:
        print(f"The '{col}' column is not present in the DataFrame.")
        
# Bar chart of mean performance by job role
job_roles = df['EmpJobRole'].unique()
mean_performance_by_role = df.groupby('EmpJobRole')['PerformanceRating'].mean()

# convert categorical attributes to numerical data types 
EmpJobRole_mapping = {'Flight Attendant': 1, 'Check-in agent': 2}
df['EmpJobRole'] = df['EmpJobRole'].map(EmpJobRole_mapping)

marital_status_mapping = {'Single': 1, 'Married': 2, 'Divorced': 3}
df['MaritalStatus'] = df['MaritalStatus'].map(marital_status_mapping)

overtime_mapping = {'Yes': 1, 'No': 0}
df['OverTime'] = df['OverTime'].map(overtime_mapping)

attrition_mapping = {'Yes': 1, 'No': 0}
df['Attrition'] = df['Attrition'].map(attrition_mapping)


# Group by mean (performance and diffrent creiteria) and print results
environment_satisfaction_means = df.groupby('EmpEnvironmentSatisfaction')['PerformanceRating'].mean()
relationship_satisfaction_means = df.groupby('EmpRelationshipSatisfaction')['PerformanceRating'].mean()
job_satisfaction_means = df.groupby('EmpJobSatisfaction')['PerformanceRating'].mean()

# Create DataFrames
environment_df = pd.DataFrame({'Environment Satisfaction': environment_satisfaction_means.index, 'Performance Rating': environment_satisfaction_means.values})
relationship_df = pd.DataFrame({'Relationship Satisfaction': relationship_satisfaction_means.index, 'Performance Rating': relationship_satisfaction_means.values})
job_df = pd.DataFrame({'Job Satisfaction': job_satisfaction_means.index, 'Performance Rating': job_satisfaction_means.values})

# Print result
print("Performance Rating change in accordance to the Environment Satisfaction:")
print(environment_df)
print("\nPerformance Rating change in accordance to the Relationship Satisfaction:")
print(relationship_df)
print("\nPerformance Rating change in accordance to the Job Satisfaction:")
print(job_df)

# Create a heatmap of the correlation matrix and Display the correlation matrix
correlation_matrix = df.corr()
print("Correlation Matrix:")
print(correlation_matrix)
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='YlOrBr', fmt=".3f", linewidths=.5, annot_kws={"size": 8})
plt.title('Correlation Matrix Heatmap')
plt.show()

# Normalize numerical columns
numerical_columns = df.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Split data into features (x) and target (y)and training and testing sets
x = df.drop(['PerformanceRating'], axis=1)
y = df['PerformanceRating']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Now it's time to make the model learn, let's use linear regression
ml = LinearRegression()
ml.fit(x_train, y_train)
y_pred = ml.predict(x_test)
predicted_value = ml.predict([[22, 1, 1, 10, 2, 4, 2, 3, 1, 18, 1, 11, 2, 5, 1]])
actual_value = df.loc[0, 'PerformanceRating']
myr2score = r2_score(y_test, y_pred)
pred_y_df = pd.DataFrame({'Actual Value': y_test, 'Predicted value': y_pred, 'Difference': y_test - y_pred})
model = sm.OLS(y, x).fit()


# Scatter plot 
plt.figure(figsize=(12, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.title('Actual vs. Linear Regression Predicted')
plt.xlabel('Actual Values')
plt.ylabel('LR Predicted Values')
plt.show()

print("This is predicted_value ", predicted_value)
print("********************************")
print("This is actual_value ", actual_value)
print("********************************")
print("This is r2score for LR evaluation ", myr2score)
print("********************************")
print(pred_y_df[0:20])
print("********************************")
print("This is model.summary2 ", model.summary2())
print("********************************")
