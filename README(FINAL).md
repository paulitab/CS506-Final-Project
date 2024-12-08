# CS506-Final-Project
# Resume Evaluation and Career Path Prediction Model


## How to Build and Run the Code
- This project is equipped with a makefile to automate setup and execution.
- Follow the steps below
  - Clone the repository to your local machine
  - Run Make Commands (the makefile provides three command options)
    - make install (install all required Python libraries and tools)
    - make run (execute the project code)
    - make all (run both make install and make run in one step)
   
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
  - Job descriptions will be collected using public APIs from various job platforms, including Indeed, GitHub Jobs, and ZipRecruiter. These platforms provide APIs that allow access to job listings, which can simplify the data collection process compared to web scraping.
  - Utilizing these APIs will enable us to retrieve structured data, including job titles, descriptions, requirements, and salary information, ensuring we have up-to-date and relevant job market insights.


- Job Descriptions
  - Job descriptions will be collected using the same approach as salary trends -- using public APIs and language processing models to accurately summarize job descriptions 
  and assign each a score based on their similarities to a candidate's resume.
  - Job requirements may also need to be collected to determine likelihood and job
    description more accurately.

## Modelling The Data
- Implement classification models (e.g. Random Forest, XGBoost) to predict the future job role or industry based on resume features.
- Build a regression model (e.g. Linear Regression) to estimate the expected salary for a candidate based on their current experience, skills, and qualifications.
- Use Natural Language Processing (NLP) methods (e.g., TF-IDF, Word Embeddings) to extract key information from resumes and compare it with job descriptions.

## NLP
  - Collect, clean and preprocess resumes and job descriptions by removing stop words, punctuation, and performing stemming or lemmatization.
  - Capture the significance of terms in the context of resumes and job listings by utilizing techniques such as TF-IDF to convert text into numerical representations. 
  - Select the specific metric for assessing similarity between job postings and resumes, such as Jaccard index, or Euclidean distance, to effectively compare textual features and quantify alignment. 

## Visualizing The Data
- Create SanKey diagrams to show potential career paths based on the resume analysis.
- Visualize salary predictions using box plots, highlighting expected salaries across different industries and experience levels.
- Present a ranked list of recommended jobs along with visual cues on how well a resume matches each job description using similarity scores.

## Testing The Data
- Withhold 20% of the data for testing purposes and use the remaining 80% for model training and validation.
- Implement cross-validation techniques to ensure the model performs well on unseen data.
- Test the model on live job postings and resumes collected during the project to evaluate the accuracy of predictions and recommendations.

