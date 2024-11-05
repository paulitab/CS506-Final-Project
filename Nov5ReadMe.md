---

# Resume Classification Model

## Project Goal
The goal of this project is to develop a predictive model that classifies resumes into one of 24 specific job categories. Given a resume as input, the model predicts the best-fitting job title, effectively aligning the candidate’s experience and skills with the most suitable role.

---

## Preliminary Data Visualization and Analysis

### Dataset Overview
Our dataset consists of **2,482 resumes** across 24 unique job categories. Initial analysis of category distribution shows minor class imbalance:
- **Least Represented Categories**: BPO (22 instances), Automobile (36 instances), and Agriculture (63 instances).
- **Most Represented Categories**: Categories range from 96 to 120 instances, indicating relatively balanced representation outside the smallest classes.

A **pie chart visualization** highlights the slight class imbalance, with BPO standing out as the most underrepresented. While this imbalance is relatively mild, predicting BPO accurately may prove challenging due to limited instances. We’ll account for this during model evaluation and consider using class-weight adjustments to address minority class performance.

---

## Data Processing and Feature Engineering

### Text Processing
- **Resume Parsing**: We focus on the `Resume_str` field, which contains the resume text in plain format, while extraneous fields (e.g., `Resume_html`) are removed for cleaner data processing.

### Feature Engineering
#### 1. **SBERT Embeddings**:
   - **Purpose**: We utilize Sentence-BERT (SBERT) to transform each resume into a dense vector of 384 dimensions. SBERT captures semantic meaning at the sentence level, making it ideal for understanding the contextual relevance of resumes for various job titles.
   - **Why SBERT?**: Unlike traditional word embeddings like Word2Vec, SBERT captures the full contextual meaning of sentences, not just individual words, which is crucial in resumes where context can vary significantly.
   - **Cosine Similarity Validation**: To validate SBERT embeddings, we calculated cosine similarity between resumes of the same category. For example, Resume 0 and Resume 1 from the same category scored a cosine similarity of 0.677, indicating a strong similarity in content and structure.

#### 2. **TF-IDF for Keyword Extraction**:
   - **Purpose**: To capture keyword frequency, we applied TF-IDF as a secondary feature extraction method. TF-IDF highlights important words that may be job-category indicators, such as higher frequencies of "Python" and "software" in tech resumes compared to arts-related resumes.
   - **Benefit**: TF-IDF provides a view into term importance by emphasizing words that appear frequently within a category while appearing infrequently across other categories, making it useful for identifying technical or domain-specific terms.

---

## Data Modeling Methods

### Model Selection

#### **Support Vector Machine (SVM)**:
   - **Rationale**: We chose SVM for its robustness with high-dimensional data, making it suitable for the 384-dimensional SBERT embeddings used here.
   - **One-vs-Rest Strategy**: Given the multi-class nature of our task, we implemented a One-vs-Rest approach, which constructs a binary classifier for each of the 24 job titles. This allows the model to make distinct predictions per class, ultimately selecting the class with the highest probability for each resume.

### Model Training and Evaluation
- **Train-Test Split**: We applied an 80:20 train-test split for evaluation. **F1 score** is used as the primary evaluation metric, which balances precision and recall, providing a clearer picture of model performance than accuracy alone. 
   - **Why F1 Score?**: F1 score is particularly valuable in imbalanced datasets where accuracy may be misleading. For example, in a cancer detection model with 90% of data labeled as "non-cancerous," a model that predicts “non-cancerous” for every instance would achieve 90% accuracy, despite being ineffective at detecting cancer cases. F1 score, by considering both precision and recall, reveals a model's true predictive capability, especially for underrepresented classes.

---

## Preliminary Results

### Results Summary
- **Initial Predictions**: Early predictions with the SVM model showed promising performance on majority classes, with F1 scores ranging from 80-90% for well-represented categories. However, the minority classes, such as Automobile (6 test samples, F1 score of 0.44) and BPO (2 test samples, F1 score of 0.5), had significantly lower F1 scores. This discrepancy highlights the model's strong performance on majority classes but points to challenges with minority class predictions.
- **Cosine Similarity Validation**: Cosine similarity between SBERT embeddings of resumes in the same category showed high similarity scores, validating that SBERT effectively captures relevant patterns for classification.
- **TF-IDF Analysis**: The TF-IDF matrix also showed promise, with resumes in the same category consistently displaying higher frequencies of specific keywords compared to those in different categories, reinforcing the utility of TF-IDF for detecting job-category-specific terms.

---

## Challenges and Next Steps

### Addressing Data Imbalance
The underrepresentation of certain classes, particularly BPO and Automobile, indicates a need for techniques to balance class performance. Future steps to address this include:
1. **Dataset Expansion**: We plan to source additional data to improve minority class representation, as the current dataset of 2,482 resumes is somewhat limiting.
2. **Class Weight Adjustments**: We’ll explore weighting options such as inverse class proportions or custom weights to enhance the model’s sensitivity to minority classes.

### Feature Expansion
Our current model uses SBERT embeddings and TF-IDF scores as features. However, we are exploring additional features to further enhance predictive accuracy and class separation.

### Adding a Job Description Matching Component
We are considering an additional component that compares resumes directly with job descriptions. This would involve developing a second model to parse job descriptions and then combining it with our current resume model. The result would allow users to paste a job description and upload their resume to receive a matching score, indicating how well the resume aligns with the specific job. This change would shift the project goal slightly from pure resume classification to resume-job matching, adding value to the model but introducing additional complexity.

---

This README provides an overview of the current state of data processing, feature engineering, modeling approaches, and preliminary results. Our next steps focus on addressing data imbalance, expanding features, and potentially implementing a job-description matching feature to increase the model’s versatility.
