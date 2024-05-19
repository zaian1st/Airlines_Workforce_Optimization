import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers
import statsmodels.api as sm


# Load the dataset
df = pd.read_csv("fau_airlines_performance.csv", sep=',')
columns_to_drop = ['Gender', 'EmpNumber']

# check for any missing values
missingdata=df.isnull().values.any()
print("This is missingdata check ", missingdata)
print("********************************")
print("This is df.columns " , df.columns)
print("********************************")


# Genral plot for no of emplyees and performance and  Display the plot
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
print("Mean Performance for Each Job Role:",mean_performance_by_role)
plt.figure(figsize=(12, 6))
mean_performance_by_role.plot(kind='bar', color=['red', 'black'])
plt.title('Mean Performance by Job Role')
plt.xlabel('Job Role')
plt.ylabel('Mean Performance Rating')
plt.xticks(range(len(job_roles)), job_roles, rotation=45, ha='right')
plt.show()

# attributes should be dropped
for col in columns_to_drop:
    if col in df.columns:
        df = df.drop([col], axis=1)
        print(f"The '{col}' column has been dropped.")
    else:
        print(f"The '{col}' column is not present in the DataFrame.")


# Map EmpJobRole to numerical values
EmpJobRole_mapping = {'Flight Attendant': 1, 'Check-in agent': 2}
df['EmpJobRole'] = df['EmpJobRole'].map(EmpJobRole_mapping)

# Map MaritalStatus to numerical values
marital_status_mapping = {'Single': 1, 'Married': 2, 'Divorced': 3}
df['MaritalStatus'] = df['MaritalStatus'].map(marital_status_mapping)

# Map OverTime to numerical values
overtime_mapping = {'Yes': 1, 'No': 0}
df['OverTime'] = df['OverTime'].map(overtime_mapping)

# Map Attrition to numerical values
attrition_mapping = {'Yes': 1, 'No': 0}
df['Attrition'] = df['Attrition'].map(attrition_mapping)

# Bar chart of mean performance by job role
job_roles = df['EmpJobRole'].unique()
mean_performance_by_role = df.groupby('EmpJobRole')['PerformanceRating'].mean()

# Group by mean (performance and diffrent creiteria) and print results
environment_satisfaction_means = df.groupby('EmpEnvironmentSatisfaction')['PerformanceRating'].mean()
relationship_satisfaction_means = df.groupby('EmpRelationshipSatisfaction')['PerformanceRating'].mean()
job_satisfaction_means = df.groupby('EmpJobSatisfaction')['PerformanceRating'].mean()
print("Performance Rating change in accordance to the Environment Satisfaction:")
print(environment_satisfaction_means)
print("\nPerformance Rating change in accordance to the Relationship Satisfaction:")
print(relationship_satisfaction_means)
print("\nPerformance Rating change in accordance to the Job Satisfaction:")
print(job_satisfaction_means)

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

# Split data into features (x) and target (y) and into training and testing sets
x = df.drop(['PerformanceRating'], axis=1)
y = df['PerformanceRating']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Neural Network
model = keras.Sequential([
    layers.Dense(128, activation='sigmoid', input_shape=(x_train.shape[1],)),
    layers.Dense(64, activation='sigmoid'),
    layers.Dense(32, activation='sigmoid'),
    layers.Dense(16, activation='sigmoid'),
    layers.Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=35, batch_size=35, validation_data=(x_test, y_test))
y_pred_nn = model.predict(x_test)
y_pred_nn = y_pred_nn.flatten()
myr2score_nn = r2_score(y_test, y_pred_nn)

print("Neural Network R2 Score:", myr2score_nn)

pred_y_df_nn = pd.DataFrame({'Actual Value': y_test, 'Predicted value': y_pred_nn, 'Difference': y_test - y_pred_nn})
print("Neural Network Predictions:")
print(pred_y_df_nn.head())

# Display the model summary 
print("********************************")
print("This is model.summary ", model.summary())
print("********************************")

# Scatter plot 
plt.figure(figsize=(12, 6))
sns.scatterplot(x=y_test, y=y_pred_nn)
plt.title('Actual vs. Neural Network Predicted')
plt.xlabel('Actual Values')
plt.ylabel('NN Predicted Values')
plt.show()


