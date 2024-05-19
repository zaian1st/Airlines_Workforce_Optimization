import pandas as pd
from pulp import *

week = ["Mon", "Tue", "Wen", "Thu", "Fri", "Sat", "Sun"]

# Read data from the Excel file
df = pd.read_excel("fau_airlines_shifts_peak.xlsx")

# Filling NaN values with 0 and converting "X" to 1
df = df.fillna(0).applymap(lambda x: 1 if x == "X" else x)

# Extracting relevant columns
avg_passenger_columns = ["Avg_Passenger_No_1", "Avg_Passenger_No_2", "Avg_Passenger_No_3",
                        "Avg_Passenger_No_4", "Avg_Passenger_No_5", "Avg_Passenger_No_6", "Avg_Passenger_No_7"]

# create a matrix to show which shift each time window is associated with
shifts = df.drop(columns=["Time Windows"] + avg_passenger_columns).values

# number of shifts
shift_num = shifts.shape[1]

# number of time windows
time_windows = shifts.shape[0]

# number of customers measured per time window
avg_customer_num = df[avg_passenger_columns].values

# service level
service_level = 1 / 32  # Setting service level as 1/25

# Decision variable, find the optimal number of workers for each time slot of each day
num_workers_indexes = []
for day_of_week in range(0, 7):  # 7 days in a week
    for shift_index in range(1, shift_num + 1):  # Adjusted shift naming
        num_workers_indexes.append(f'{week[day_of_week]} - Shift{shift_index}')

num_workers = LpVariable.dicts("num_workers", num_workers_indexes, lowBound=0, cat="Integer")
print(num_workers)

# Create problem
# Minimize the number of workers/costs paid for employees each day
prob = LpProblem("scheduling_workers", LpMinimize)

for day_of_week in range(0, 7):  # 7 days in a week
    prob += lpSum([num_workers[f'{week[day_of_week]} - Shift{j}'] for j in range(1, shift_num + 1)])  # Adjusted shift naming

# The average number of customers in each time slot must also be satisfied
for day_of_week in range(0, 7):  # 7 days in a week
    for t in range(time_windows):
        prob += lpSum([shifts[t, j - 1] * num_workers[f'{week[day_of_week]} - Shift{j}'] for j in range(1, shift_num + 1)]) >= \
                avg_customer_num[t][day_of_week] * service_level

prob.solve()
print("Status:", LpStatus[prob.status])

for index in num_workers:
    index_parts = index.split(' - ')
    day = index_parts[0]
    shift = index_parts[1]
    print(f"The number of workers needed for {day} - {shift} is {int(num_workers[index].value())}")
