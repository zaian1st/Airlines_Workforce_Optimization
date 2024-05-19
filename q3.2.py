#importing necessary libraries.
import pandas as pd
import numpy as np

# import TfidfVector from sklearn.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

employees = pd.read_csv(r"fau_airlines_onboarding.csv")
mycolumns=employees.columns
print('The columns')
print(mycolumns)


def create_soup(x):
    return ''.join(x['study degree']) + ''.join(x['previous_experience']) + '' + ''.join(x['civil_status']) + '' + ''.join(x['personality_traits']) + ''.join(x['hobbies']) + ''.join(x['favourite sport'])
employees['soup'] = employees.apply(create_soup, axis=1)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(employees['soup'])
mymatrix=tfidf_matrix.shape

print(mymatrix)

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# construct a reverse map of indices and employee IDs
indices = pd.Series(employees.index, index=employees['id']).drop_duplicates()

def get_recommendations(id, cosine_sim=cosine_sim):
    
    # get the index of the employee that matches the employee ID
    IDx = indices[id]
    
    # get the pairwise similarity scores of all employees with the specified employee ID
    sim_scores = list(enumerate(cosine_sim[IDx]))
    
    # sort employees based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # get the scores of the three most similar employees
    sim_scores = sim_scores[1:4]
    
    # get employee indices
    employees_indices = [i[0] for i in sim_scores]
    
    # return the top three most similar employees
    return employees['id'].iloc[employees_indices]

recommendations1 = get_recommendations('emp_100', cosine_sim)
print("this is recommendations for emp_100")
print(recommendations1)
print("*********************************************************")
recommendations2 = get_recommendations('emp_101', cosine_sim)
print("this is my self as emp_101")
print(recommendations2)


