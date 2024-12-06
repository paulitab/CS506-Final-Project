### **Data Processing and Feature Engineering**

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

---

### **Modeling**

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

### **Hyperparameter Tuning**

The following hyperparameter grid was explored during GridSearchCV:
- **`C`**: Inverse of regularization strength (tested values: `[0.01, 0.1, 1, 10, 100]`).
- **`penalty`**: Regularization type (`'l2'` tested).
- **`solver`**: Optimization solver (`'saga'` and `'lbfgs'` tested).
- **`class_weight`**: Handling class imbalance (`None` and `'balanced'` tested).

The best parameters identified were:
```python
Best Parameters: {'C': 0.01, 'class_weight': 'balanced', 'penalty': 'l2', 'solver': 'saga'}
```

#### **Why These Parameters Work Well**:
1. **`C=0.01`**:
   - Stronger regularization (`C` is the inverse of regularization strength) prevents overfitting, especially in high-dimensional data.
2. **`class_weight='balanced'`**:
   - Adjusts for the class imbalance by assigning higher weights to underrepresented classes, improving recall and F1-scores for these categories.
3. **`solver='saga'`**:
   - Optimized for large datasets and supports `l2` regularization with high-dimensional sparse data.

The tuned model achieved significant improvements, especially in F1-scores for minority classes, compared to the baseline Logistic Regression model.

---

### **Results**

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
