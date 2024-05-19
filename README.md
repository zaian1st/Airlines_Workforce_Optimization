# Airline Workforce Optimization

Welcome to the Airline Workforce Optimization project! This repository contains a comprehensive analysis and solution for optimizing the workforce planning and performance at an airline.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset and Preprocessing](#dataset-and-preprocessing)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Machine Learning Models](#machine-learning-models)
   - [Employee Performance Prediction](#employee-performance-prediction)
   - [Recommender System](#recommender-system)
5. [Results and Insights](#results-and-insights)
6. [How to Run the Code](#how-to-run-the-code)
7. [Conclusion](#conclusion)
8. [Contact](#contact)

## Project Overview

The airline is aiming to improve its workforce planning to ensure efficiency and satisfaction among employees. This project includes an in-depth analysis of employee data, the development of predictive models, and the creation of a recommender system to enhance employee onboarding and performance.

## Dataset and Preprocessing

The dataset used in this project contains various attributes related to employee demographics, job roles, satisfaction levels, and performance ratings. Preprocessing steps include handling missing values, encoding categorical variables, and normalizing numerical features.

## Exploratory Data Analysis (EDA)

EDA helps in understanding the underlying patterns and relationships in the data. Key insights from the EDA include:

- Distribution of performance ratings.
- Mean performance based on job roles.
- Correlation analysis of different factors affecting performance.

*Relevant scripts:*
- [EDA Script](./q1.4.py)
- [EDA Peak Analysis](./q1.4_peak.py)

## Machine Learning Models

### Employee Performance Prediction

To predict future employee performance, we developed two models:

1. **Neural Network Model**: A deep learning model using Keras to predict performance ratings based on various features.
2. **Linear Regression Model**: A traditional regression approach to understand the linear relationships between features and performance.

*Relevant scripts:*
- [Neural Network Model](./q3.4_neuralN.py)
- [Linear Regression Model](./q3.4lg.py)

### Recommender System

A recommender system was created to connect new hires with existing employees who share similar interests and characteristics, enhancing socialization and onboarding efficiency.

*Relevant script:*
- [Recommender System](./q3.2.py)

## Results and Insights

The analysis provided several insights:
- Key factors influencing employee performance include job satisfaction, job role, and work-life balance.
- The neural network model slightly outperforms the linear regression model with an R² score of 0.644.
- The recommender system successfully identified employees with similar profiles to new hires, promoting better onboarding experiences.

## How to Run the Code

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/Airline_Workforce_Optimization.git
   cd Airline_Workforce_Optimization
2. Install the required libraries:
   Pandas: For data manipulation and analysis.
      NumPy: For numerical computations.
      Matplotlib: For creating static, animated, and interactive visualizations.
      Seaborn: For statistical data visualization.
      Scikit-learn: For machine learning algorithms and tools, including linear regression.
      Keras: For building and training neural network models.
      TensorFlow: As a backend for Keras to train the neural network.
   
## Algorithms
Linear Regression: Used for predicting employee performance based on linear relationships between features.
Neural Network: A deep learning model for predicting employee performance, which captures complex relationships in the data.
Cosine Similarity: Used in the recommender system to find similarities between employees based on their profiles.


## Conclusion
This project demonstrates the application of data analytics and machine learning to optimize workforce planning and improve employee performance at the airline. The integration of predictive models and a recommender system provides actionable insights and practical solutions for HR management.

