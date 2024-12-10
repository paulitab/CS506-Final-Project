import pytest
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Mock data for testing
resumes = pd.DataFrame({
    "Resume_str": [
        "Experienced software engineer with Python and Java skills.",
        "Certified project manager experienced in leading IT projects.",
        "Data scientist proficient in Python, R, and SQL."
    ],
    "Category": ["IT", "Management", "Data Science"]
})

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

@pytest.fixture
def embeddings():
    """Fixture to generate embeddings for testing."""
    return resumes['Resume_str'].apply(lambda x: model.encode(x)).tolist()

def test_embedding_generation(embeddings):
    """Test that embeddings are generated and have correct dimensions."""
    print("Running test: test_embedding_generation")
    assert len(embeddings) == len(resumes), "Embedding count does not match resume count."
    assert all(isinstance(embedding, np.ndarray) for embedding in embeddings), "Not all embeddings are numpy arrays."
    assert all(embedding.shape == (384,) for embedding in embeddings), "Embedding dimensions are incorrect."
    print("PASSED: test_embedding_generation")

def test_cosine_similarity(embeddings):
    """Test cosine similarity calculation between two embeddings."""
    print("Running test: test_cosine_similarity")
    similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    assert 0 <= similarity_score <= 1, "Cosine similarity score is out of bounds."
    print("PASSED: test_cosine_similarity")

def test_feature_engineering():
    """Test feature engineering (e.g., resume length)."""
    print("Running test: test_feature_engineering")
    resumes['Resume_Length'] = resumes['Resume_str'].apply(len)
    assert 'Resume_Length' in resumes.columns, "Resume_Length column not found."
    assert all(isinstance(length, int) for length in resumes['Resume_Length']), "Resume lengths are not integers."
    assert all(length > 0 for length in resumes['Resume_Length']), "Some resumes have zero length."
    print("PASSED: test_feature_engineering")

def test_data_integrity():
    """Test that there are no missing values in the dataset."""
    print("Running test: test_data_integrity")
    assert resumes.isnull().sum().sum() == 0, "Dataset contains missing values."
    print("PASSED: test_data_integrity")

def test_category_specific_keyword_counts():
    """Test category-specific keyword counts (if implemented)."""
    print("Running test: test_category_specific_keyword_counts")
    # Mock a keyword dictionary and feature
    keyword_dict = {"Python": "IT", "Java": "IT", "SQL": "Data Science"}
    resumes['Keyword_Counts'] = resumes['Resume_str'].apply(
        lambda x: sum([1 for word in x.split() if word in keyword_dict])
    )
    assert 'Keyword_Counts' in resumes.columns, "Keyword_Counts column not found."
    assert all(isinstance(count, int) for count in resumes['Keyword_Counts']), "Keyword counts are not integers."
    print("PASSED: test_category_specific_keyword_counts")

def test_calculate_scores():
    """Test the calculate_scores function to ensure it calculates category scores correctly."""
    print("Running test: test_calculate_scores")
    # Mock data
    text = "Experienced Python developer with skills in Java and SQL."
    keywords_dict = {
        "IT": ["Python", "Java", "SQL"],
        "Finance": ["Accounting", "Excel", "Budget"],
        "Management": ["Leadership", "Strategy", "Project"],
    }

    # Expected output
    expected_scores = [3, 0, 0]  # Matches: 3 in IT, 0 in Finance, 0 in Management

    # Call the function
    scores = calculate_scores(text, keywords_dict)

    # Assertions
    assert isinstance(scores, list), "Output should be a list."
    assert len(scores) == len(keywords_dict), "Output list length should match number of categories."
    assert scores == expected_scores, f"Expected scores {expected_scores}, but got {scores}."
    print("PASSED: test_calculate_scores")

# Mocked calculate_scores function
def calculate_scores(text, keywords_dict):
    scores = []
    for category, keywords in keywords_dict.items():
        score = sum(text.lower().count(keyword.lower()) for keyword in keywords)
        scores.append(score)
    return scores

if __name__ == "__main__":
    pytest.main(["-s"])  # Run pytest with -s to see print statements
