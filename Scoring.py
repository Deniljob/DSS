import pandas as pd
import numpy as np

# Simulated Extended Data
data = {
    'School': [
        'School A', 'School B', 'School C', 'School D',
        'School E', 'School F', 'School G', 'School H'
    ],
    'Test_Scores': [85, 90, 78, 88, 84, 79, 91, 76],  # out of 100
    'Graduation_Rate': [92, 95, 88, 96, 90, 85, 97, 80],  # percentage
    'Attendance': [95, 97, 93, 98, 94, 92, 96, 89],  # percentage
    'Funding_Per_Student': [1000, 1100, 950, 1050, 1020, 980, 1080, 920],  # dollars
    'Staffing_Levels': [50, 55, 45, 60, 52, 48, 58, 43],  # number of staff
    'Material_Availability': [85, 90, 80, 95, 88, 82, 92, 78],  # out of 100
    'Socioeconomic_Status': [30, 20, 50, 10, 40, 60, 15, 45],  # percentage qualifying for free/reduced lunch
    'Special_Education_Needs': [15, 10, 20, 5, 18, 22, 8, 25],  # number of students
    'Language_Proficiency': [25, 20, 30, 15, 28, 35, 18, 40]  # percentage of English learners
}

# Create a DataFrame
df = pd.DataFrame(data)

# Data Cleaning: Remove duplicates, fix errors, and handle missing values
df.drop_duplicates(inplace=True)

# Let's introduce some missing values for demonstration
df.loc[3, 'Test_Scores'] = np.nan
df.loc[5, 'Graduation_Rate'] = np.nan

# Handling Missing Values: Mean imputation for simplicity
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].apply(lambda x: x.fillna(x.mean()))

# Normalization: Min-Max Scaling
def min_max_scaling(series):
    return (series - series.min()) / (series.max() - series.min())

for column in df.columns[1:]:
    df[column] = min_max_scaling(df[column])

# Define weights for each criterion
weights = {
    'Test_Scores': 0.2,
    'Graduation_Rate': 0.2,
    'Attendance': 0.15,
    'Funding_Per_Student': 0.1,
    'Staffing_Levels': 0.1,
    'Material_Availability': 0.1,
    'Socioeconomic_Status': 0.05,
    'Special_Education_Needs': 0.05,
    'Language_Proficiency': 0.05
}

# Function to calculate the weighted score for each school
def calculate_weighted_score(row, weights):
    score = sum(row[criterion] * weight for criterion, weight in weights.items())
    return score

# Calculate weighted scores for all schools
df['Weighted_Score'] = df.apply(calculate_weighted_score, axis=1, weights=weights)

# Rank schools based on weighted scores
df['Rank'] = df['Weighted_Score'].rank(ascending=False)

# Sort the DataFrame by rank
df_sorted = df.sort_values('Rank')

# Display the sorted DataFrame
print(df_sorted)

# Output the DSS recommendations
def dss_recommendation(df):
    print("Decision Support System Recommendations for Resource Allocation:\n")
    for index, row in df.iterrows():
        print(f"Rank {int(row['Rank'])}: {row['School']} with a score of {row['Weighted_Score']:.2f}")

# Run the DSS recommendation function
dss_recommendation(df_sorted)
