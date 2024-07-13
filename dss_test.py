# test_dss.py
import pytest
import pandas as pd
import numpy as np
from dss import load_data, clean_data, normalize_data, rank_schools, dss_recommendation

@pytest.fixture
def sample_data():
    df = load_data()
    return df

@pytest.fixture
def weights():
    return {
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

def test_load_data(sample_data):
    assert isinstance(sample_data, pd.DataFrame)
    assert len(sample_data) == 8

def test_clean_data(sample_data):
    cleaned_df = clean_data(sample_data)
    assert cleaned_df.isna().sum().sum() == 0

def test_normalize_data(sample_data):
    cleaned_df = clean_data(sample_data)
    normalized_df = normalize_data(cleaned_df)
    for column in normalized_df.columns[1:]:
        assert normalized_df[column].min() == 0
        assert normalized_df[column].max() == 1

def test_rank_schools(sample_data, weights):
    cleaned_df = clean_data(sample_data)
    normalized_df = normalize_data(cleaned_df)
    ranked_df = rank_schools(normalized_df, weights)
    assert 'Weighted_Score' in ranked_df.columns
    assert 'Rank' in ranked_df.columns
    assert ranked_df['Rank'].min() == 1
    assert ranked_df['Rank'].max() == 8

def test_dss_recommendation(sample_data, weights):
    cleaned_df = clean_data(sample_data)
    normalized_df = normalize_data(cleaned_df)
    ranked_df = rank_schools(normalized_df, weights)
    recommendations = dss_recommendation(ranked_df)
    assert len(recommendations) == 8
    assert "Rank 1:" in recommendations[0]
