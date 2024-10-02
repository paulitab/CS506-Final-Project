# CS506-Final-Project
# Resume Evaluation and Career Path Prediction Model

## Project Description
This project aims to develop a machine learning model that evaluates resumes. Typically, large companies do not have enough time to open each CV, so they use machine learning algorithms for the Resume Screening task. We want to take this a step beyond and make it so that users can Pre-Screen their Resume and see future possible job titles and salaries bases on their resume. The model will analyze key features such as education, work experience, and skills to forecast future job roles, industries, and salary trajectories. 

## Goals
- Identify and compare common career paths across different industries.
- Successfully predict the likely next job roles or industries based on a user's current experience, skills, and education.
- Use regression techniques to estimate the future salary a user can expect based on their resume features.
- Analyze how different features such as age or years of experience impact career prospects.

## Data To Be Collected
- Resume Data
  - Collect resume data (text) from public sources such as Kaggle (below)
  - The dataset will contain features such as job title, skills, education, years of experience, and salary (depending on available datasets).
  - https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset
 
- Salary Trends
  - Scrape job market data from online platforms like Glassdoor, LinkedIn Salary
    Insights, or Bureau of Labor Statistics to gather information about current job
    trends and salary benchmarks.

- Job Descriptions
  - Job descriptions will be gathered through scraping or using available datasets to
    match resumes with the latest job opportunities.
  - Job requirements may also need to be collected to determine likelihood and job
    description more accurately.

## Modelling The Data
- Implement classification models (e.g. Random Forest, XGBoost) to predict the future job role or industry based on resume features.
- Build a regression model (e.g. Linear Regression) to estimate the expected salary for a candidate based on their current experience, skills, and qualifications.
- Use Natural Language Processing (NLP) methods (e.g., TF-IDF, Word Embeddings) to extract key information from resumes and compare it with job descriptions.

## Visualizing The Data
- Create SanKey diagrams to show potential career paths based on the resume analysis.
- Visualize salary predictions using box plots, highlighting expected salaries across different industries and experience levels.
- Present a ranked list of recommended jobs along with visual cues on how well a resume matches each job description using similarity scores.

## Testing The Data
- Withhold 20% of the data for testing purposes and use the remaining 80% for model training and validation.
- Implement cross-validation techniques to ensure the model performs well on unseen data.
- Test the model on live job postings and resumes collected during the project to evaluate the accuracy of predictions and recommendations.
