# Resume Classification Model

## Goal
The goal of this project is to develop a predictive model that can classify a resume into one of 24 specific job categories. Given a resume as input, the model will predict the best-fitting job title, effectively matching the candidate's background to the most suitable role.

---

## Preliminary Visualizations of Data

### Dataset Overview
The dataset consists of 2,482 resumes categorized into 24 unique job titles. The distribution of these categories reveals a marginal data imbalance:
- **Least Represented Categories**: BPO with 22 instances, Automobile with 36 instances, and Agriculture with 63 instances.
- **Most Represented Categories**: The majority of categories have between 96 and 120 instances.
  
A pie chart visualization of the category distribution shows that while some imbalance exists, it is relatively mild, with BPO standing out as the most underrepresented. This distribution will be considered during model evaluation, as predicting the BPO category could present challenges due to its underrepresentation.

---

## Data Processing

### Text Processing
- **Resume Parsing**: We specifically use the `Resume_str` field for modeling. Irrelevant columns, such as `Resume_html`, are removed to streamline processing.
- **Text Normalization**: Basic preprocessing techniques (e.g., tokenization, stop word removal) are applied, and the text is formatted for input into NLP models.

### Feature Engineering
#### 1. **SBERT Embeddings**:
   - **Purpose**: SBERT (Sentence-BERT) is used to represent each resume as a dense vector of 384 dimensions. This embedding captures the semantic meaning of sentences and provides a context-aware representation of the resume’s content.
   - **Why SBERT?**: SBERT is preferred over traditional embeddings (e.g., Word2Vec) as it captures the full meaning of sentences, not just individual words, which aligns well with the nature of resumes where context is crucial.
   - **Example Analysis**: To check the effectiveness of SBERT embeddings, cosine similarity was calculated between resumes of the same category. For example, resume 0 and resume 1 from the same category had a cosine similarity score of 0.677, indicating strong similarity in sentence structure and content.
   
#### 2. **TF-IDF for Keyword Extraction**:
   - **Purpose**: In addition to SBERT, we plan to use TF-IDF to extract keyword frequencies as another feature. This approach captures important keywords that may serve as indicators of specific job categories.
   - **Benefit**: TF-IDF provides insights into term importance, allowing the model to understand word frequencies and highlight relevant keywords. This is particularly useful for identifying technical or domain-specific terms, like "software" in tech resumes.

---

## Modeling Methods

### Model Selection
#### **Support Vector Machine (SVM)**:
   - **Rationale**: SVM is chosen for its effectiveness with high-dimensional data, such as the 384-dimensional SBERT embeddings.
   - **One-vs-Rest Strategy**: Given the multi-class nature of the task, a One-vs-Rest approach is implemented, which builds binary classifiers for each of the 24 job titles. This approach allows the SVM to efficiently differentiate between job categories.

### Model Training and Evaluation
- **Train-Test Split**: We divided the data into training and testing sets to evaluate model performance. Initial experiments used an F1 score for evaluation, with plans to address data imbalance by focusing on metrics like weighted F1 score to ensure fair representation across all job categories.

---

## Preliminary Results

### Results Summary
- **Initial Predictions**: Early predictions based on the SVM model showed promising results on the majority classes, although further tuning is required to improve predictions on underrepresented categories like BPO.
- **Cosine Similarity Validation**: Cosine similarity between SBERT embeddings for resumes within the same category showed high similarity scores, confirming that the SBERT embeddings effectively capture category-specific patterns.
  
### Challenges and Next Steps
- **Data Imbalance**: The slight imbalance, especially for BPO and Automobile categories, indicates a need to address minority class performance through techniques like class-weight adjustments in the SVM model or data augmentation.
- **Feature Combination**: Integrating TF-IDF keywords with SBERT embeddings may improve classification accuracy by enhancing keyword recognition alongside semantic understanding.

---

This README outlines the current state of data processing, feature engineering, modeling, and preliminary findings, setting a clear path for ongoing improvements and final model development.
