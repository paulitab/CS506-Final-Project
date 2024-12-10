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
This project aims to develop a machine learning model that evaluates resumes. Typically, large companies do not have enough time to open each CV, so they use machine learning algorithms for the Resume Screening task. We want to take this a step beyond and make it so that users can Pre-Screen their Resume and see future possible job titles. The model will analyze key features such as education, work experience, and skills to forecast future job roles and industries. 

## Goals
- Identify and compare common career paths across different industries.
- Successfully predict the likely next job roles or industries based on a user's current experience, skills, and education.
- Analyze how different features such as keywords on resume or resume length relate to specifc jobs
- Handle data imbalances

## Data To Be Collected
- Resume Data
  - Collect resume data (text) from public sources such as Kaggle (below)
  - The dataset will contain features such as resume_id, resume_decription, resume_html, and category
  - https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

## Data Processing and Feature Engineering

We began with two datasets: the **resume dataset** and the **job description dataset**. After evaluating both, we decided to exclusively use the **resume dataset**, as it was more extensive and mixing the datasets resulted in reduced model performance. The original resume dataset included the following columns: 

`Index(['ID', 'Resume_str', 'Resume_html', 'Category'], dtype='object')`.

To streamline the data for modeling, we removed the `Resume_html` column because it was essentially a redundant HTML representation of `Resume_str`. The remaining features (`ID`, `Resume_str`, and `Category`) were considered essential for prediction.

In the **feature engineering stage**, we created additional features to enhance the predictive accuracy of the model:
1. **`resume_length`**:
   - This feature captures the length of each resume string.
2. **`SentenceTransform`**:
   - A multi-dimensional matrix (384 dimensions) generated from the SentenceTransformer model. This embedding represents the semantic meaning of the resume text in a mathematical form.
3. **`Category_score`**:
   - A 24-dimensional array derived using a frequency-based keyword dictionary. Each entry in the array represents the count of category-specific keywords present in the resume.

The enriched dataset provided a well-rounded input for our predictive model, combining basic metadata (resume length), semantic embeddings, and category-specific word counts.

## Modeling

To identify the most suitable model for our **high-dimensional dataset**, we evaluated the following algorithms:
1. **RandomForest**:
   - Performance: Achieved **0.68 accuracy** with default parameters.
   - Strengths: Effective for non-linear relationships and mixed feature types.
   - Limitations: While it performed reasonably well, it struggled to capture relationships in very high-dimensional data, likely due to sparse or redundant features in the 384+ dimensions.

2. **SVC (Support Vector Classifier)(Was determined to computationally expensive to run)**:
   - **RBF kernel**:
     - While the RBF kernel is effective for non-linear problems, it was computationally expensive and impractical for our high-dimensional data.
   - **Linear kernel**:
     - Designed for high-dimensional data, but training time was prohibitively long due to the complexity introduced by combining embeddings, word counts, and additional features.
     
3. **Logistic Regression**:
   - Performance: Achieved **0.70 accuracy** with default parameters, outperforming RandomForest.
   - Strengths: Logistic Regression handles high-dimensional data efficiently when properly regularized. The linear nature of the embeddings made it well-suited for this dataset.
   - Chosen Model: Its higher accuracy and faster convergence compared to SVC and RandomForest made Logistic Regression the preferred choice for further optimization.

4. **XGBoost(Was determined to computationally expensive to run)**:
   - Performance: Poor convergence due to the very high-dimensional data, resulting in long training times and suboptimal results.

After identifying Logistic Regression as the best-performing model, we conducted **hyperparameter tuning** using **GridSearchCV** to optimize its performance.

---

## Hyperparameter Tuning

The following hyperparameter grid was explored during GridSearchCV:
- **`C`**: Inverse of regularization strength (tested values: `[0.01, 0.1, 1, 10, 100]`).
- **`penalty`**: Regularization type (`'l2'` tested).
- **`solver`**: Optimization solver (`'saga'` and `'lbfgs'` tested).
- **`class_weight`**: Handling class imbalance (`None` and `'balanced'` tested).

The best parameters identified were:
```python
Best Parameters: {'C': 0.01, 'class_weight': 'balanced', 'penalty': 'l2', 'solver': 'saga'}
```

## Why These Parameters Work Well:
1. **`C=0.01`**:
   - Stronger regularization (`C` is the inverse of regularization strength) prevents overfitting, especially in high-dimensional data.
2. **`class_weight='balanced'`**:
   - Adjusts for the class imbalance by assigning higher weights to underrepresented classes, improving recall and F1-scores for these categories.
3. **`solver='saga'`**:
   - Optimized for large datasets and supports `l2` regularization with high-dimensional sparse data.

The tuned model achieved significant improvements, especially in F1-scores for minority classes, compared to the baseline Logistic Regression model.

---

## Results

#### **Classification Report**
The final classification report highlighted key improvements:
1. **Overall Accuracy**: Increased from **0.70** to **0.73**.
2. **Macro Average F1-Score**: Increased from baseline values, indicating better performance across all classes, including underrepresented ones.
3. **Underrepresented Classes**:
   - Categories like `AGRICULTURE`, `BPO`, and `ARTS`, which initially had poor recall and F1-scores, showed marked improvement due to the use of `class_weight='balanced'`.

#### **Confusion Matrix**
The confusion matrix visualizes the true vs. predicted labels:
1. **Diagonal Dominance**:
   - Most predictions align along the diagonal, indicating correct classifications.
2. **Improvements for Minority Classes**:
   - Underrepresented categories (e.g., `AGRICULTURE`, `AUTOMOBILE`) show fewer misclassifications compared to the baseline model.

---

### **Key Takeaways**
- **Logistic Regression** emerged as the best-performing model for our high-dimensional dataset due to its efficiency and compatibility with the semantic structure of SentenceTransformer embeddings.
- **Hyperparameter Tuning** with class balancing significantly improved performance for minority classes, addressing the challenge of data imbalance effectively.
- The combination of optimized features (resume length, semantic embeddings, and category-specific word counts) and a well-tuned Logistic Regression model provided a robust solution for resume category classification.


## Analysis of Visualizations
Note: Most of these visualizations are interactive so please look at jupyter notebook to see the interactive. 
Visualization #1

![c905cbd5-6c12-4986-b6ef-704920e486a3](https://github.com/user-attachments/assets/37314fc6-b7bf-42c0-9c39-31dad0ffff02)

- This graph shows us a confusion matrix of the different jobs and how accurately we predicted them with our model.
- Some jobs had high accuracy predictions, such as Accountant, Aviation, Banking, Construction, and Information-Technology.
- Some jobs had a wider spread regarding predictions such as Apparel, Advocate, and Public-Relations.
- And some jobs had low accuracy of predictions, like Agriculture and Arts. 

Visualization #2

<img width="851" alt="Screen Shot 2024-12-08 at 2 26 06 PM" src="https://github.com/user-attachments/assets/f63db8a9-535d-4a48-a802-f53781be1766">

- This graph shows us a histogram of resume lengths from the database
- Most resumes have lengths between 500 and 1500 words
- The peak frequency is at around 1000 words, meaning most resumes in the dataset have a length of around 1000.
- The distribution is rightly-skewed, meaning there are fewer resumes with very long lengths compared to shorter ones.

Visualization #3
<img width="1018" alt="Screenshot 2024-12-09 at 7 17 26 PM" src="https://github.com/user-attachments/assets/653c5999-7531-40fb-a2f7-507675d5c510">

- This graph shows us how us the average length for each job category. 
- Some things to point out are that IT for example on average has 200 more words compared to Sales. 
- The job lengths could indicate specifc jobs which in return could tell help with making a better predictive model. 

Visualization #4

![5a08fd32-7d0e-4844-bb59-3c9c9ac9d716](https://github.com/user-attachments/assets/1851c6e1-09f8-4295-9c5f-16413081a3b8)
![7dba9983-5b7d-423e-9309-1779fe4a4845](https://github.com/user-attachments/assets/102fe83e-f9e8-4164-9ba1-e40d097ea42f)
![aceb37ef-44f5-48b7-a3e7-0e600d97ffd8](https://github.com/user-attachments/assets/ac23f5e3-0838-486c-91f0-53982bea9b32)

- The provided word clouds represent key skills or terms commonly associated with three distinct domains: Information Technology, Culinary Arts, and Agriculture.
- For the Information Technology, we will point out that the largest words are different programming languages, which aligns with this jobs importants in the IT domain.
- For the Culinary Arts, the largest words are words that emphasize the core skills needed in designing and executing culinary offerings. This makes sense since these are the people responsible for creating a beautiful mix of creative, technical, and managerial skills.
- The third image here shows the words relevant for Agriculture. They are the words that indicate a focus on agricultural productivity and technical expertise in farm equipment.
- This clearly shows us that each job category has unique and frequently used words for specific categories which is a useful feature to have. 



