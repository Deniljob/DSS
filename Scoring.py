import pandas as pd

data = {
    'School': [
        'School A', 'School B', 'School C', 'School D',
        'School E', 'School F', 'School G', 'School H'
    ],
    'Performance_Level': [75, 85, 65, 90, 80, 70, 95, 60],  # out of 100
    'Resource_Deficit': [30, 20, 50, 10, 40, 60, 15, 45],   # out of 100
    'Student_Demographics': [60, 70, 50, 80, 55, 65, 75, 45], # out of 100
    'Infrastructure_Quality': [70, 85, 65, 90, 75, 60, 80, 50] # out of 100
}

df = pd.DataFrame(data)

weights = {
    'Performance_Level': 0.3,
    'Resource_Deficit': 0.25,
    'Student_Demographics': 0.2,
    'Infrastructure_Quality': 0.25
}

def normalize(df, column):
    return (df[column] - df[column].min()) / (df[column].max() - df[column].min())

for column in weights.keys():
    df[column] = normalize(df, column)

def calculate_weighted_score(row, weights):
    score = sum(row[criterion] * weight for criterion, weight in weights.items())
    return score

df['Weighted_Score'] = df.apply(calculate_weighted_score, axis=1, weights=weights)

df['Rank'] = df['Weighted_Score'].rank(ascending=False)

df_sorted = df.sort_values('Rank')

print(df_sorted)

def dss_recommendation(df):
    print("Decision Support System Recommendations for Resource Allocation:\n")
    for index, row in df.iterrows():
        print(f"Rank {int(row['Rank'])}: {row['School']} with a score of {row['Weighted_Score']:.2f}")


dss_recommendation(df_sorted)
