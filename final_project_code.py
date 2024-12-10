#!/usr/bin/env python
# coding: utf-8

# ## Final Project ##

# Firstly we load the data

# In[1]:


"""
/usr/local/bin/python3.10 -m pip install pandas scikit-learn keras matplotlib sentence-transformers

"""
import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("System path:", sys.path)


# pip install datasets

# In[5]:


import pandas as pd


# Replace "Saba06huggingface/resume_dataset" 
resumes = pd.read_csv("data/resume.csv")

# splits = {'train': 'train_dataset.json', 'validation': 'val_dataset.json', 'test': 'test_dataset.json'}
# df = pd.read_json("hf://datasets/Saba06huggingface/resume_dataset/" + splits["train"], lines=True)

print(resumes.columns)

# job descirption data set form data folder
job_descriptions = pd.read_csv("data/jb_df.csv")

print(job_descriptions.columns)




# **Exploring the data**

# In[6]:


print(resumes.head())
print()
print(job_descriptions.head())


# In[7]:


print(resumes.isnull().sum())
print()
print(job_descriptions.isnull().sum())


# In[8]:


print(resumes.shape)
print(resumes.info())
print()
print(job_descriptions.shape)
print(job_descriptions.info())


# In[9]:


print(resumes['Category'].value_counts())
print()
# print the total number of resumes
print("total number of resumes:", resumes['Category'].value_counts().sum())

#Print number of unique categories 
print("total number of categories:", resumes['Category'].nunique())


# In[10]:


resumes['resume_length'] = resumes['Resume_str'].apply(lambda x: len(x.split()))
job_descriptions['job_desc_length'] = job_descriptions['Job Description'].apply(lambda x: len(x.split()))

print(resumes['resume_length'].describe())
print(job_descriptions['job_desc_length'].describe())



# In[11]:


import matplotlib.pyplot as plt

# Calculate the frequency of each category
category_counts = resumes['Category'].value_counts()

# Plot a pie chart
plt.figure(figsize=(10, 6))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Resume Categories')
plt.show()


# Second Visualization: **Word Cloud**

# In[26]:


from collections import Counter
from wordcloud import WordCloud

# Word frequency visualization for the IT category
# words taken from the dictionary created further below
category_words = ["Python", "Java", "C++", "JavaScript", "HTML", "CSS", "SQL", "NoSQL", "React", "Angular",
        "Django", "Flask", "Git", "AWS", "Azure", "Google Cloud Platform", "DevOps", "Docker",
        "Kubernetes", "Linux", "Windows Server", "Machine Learning", "Data Science", "TensorFlow",
        "Keras", "Cybersecurity", "Penetration Testing", "Scrum", "Agile", "REST APIs",
        "Microservices", "Networking", "Cloud Computing", "Virtualization", "Data Analysis"]
word_counts = Counter(category_words)

# Word cloud
wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_counts)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Word frequency visualization for the Chef category
category_words = ["Menu Planning", "Food Preparation", "Culinary Techniques", "Recipe Development",
        "Inventory Management", "Food Safety", "Sanitation Standards", "Pastry Skills",
        "Grill Station", "Sous Vide Cooking", "Knife Skills", "Garnishing", "Plating",
        "Baking", "Kitchen Operations", "Staff Management", "Cost Control",
        "Menu Costing", "Customer Service", "Catering"]

word_counts2 = Counter(category_words)

# Word cloud
wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_counts2)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Word frequency visualization for the Agriculture category
category_words = ["Crop Management", "Farm Equipment", "Horticulture", "Soil Testing", "Irrigation Systems",
        "Pest Control", "Agronomy", "Livestock Management", "Sustainable Farming", "Organic Certification",
        "Farm Operations", "Yield Optimization", "Composting", "Agricultural Policy", "Crop Rotation"]

word_counts3 = Counter(category_words)

# Word cloud
wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_counts3)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# Third Visualization: **Resume Length Distribution**

# In[30]:


# Resume Lengths Calcualtion
resumes['resume_length'] = resumes['Resume_str'].apply(lambda x: len(x.split()))

# Plot a histogram of resume lengths
plt.figure(figsize=(10, 6))
plt.hist(resumes['resume_length'], bins=30, color='skyblue')
plt.xlabel('Resume Length')
plt.ylabel('Frequency')
plt.title('Distribution of Resume Lengths')
plt.show()



# In[27]:


#  print resume_str for the first resume
print(resumes['Resume_str'][0])


# In[13]:


# drop html column from resume
resumes.drop('Resume_html', axis=1, inplace=True)


# In[14]:


#This shows the structure of the resume format currently which includes an ID, Resume_str, and a Category 
print(resumes.columns)


# In[15]:


#This shows what the values of the job_description looks like 

print(job_descriptions['Job Description'][0])
print(job_descriptions['Role'][0])
print(job_descriptions['Job Title'][0])


# 
# This section uses the Sentence Transformer with the all-MiniLM model to map text into high-dimensional vectors (essentially numerical matrices). 
# These vectors allow us to compare different pieces of text in a mathematical way. 
# For example, sentences like 'It's a sunny day' and 'It's a warm day and the sun is out' will have similar vectors because they mean similar things.

# Continuing with resume dataset from here onwards

# In[16]:


# pip install -U sentence-transformers
from sentence_transformers import SentenceTransformer


# Load the SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example: Encode a single resume and a job description
resume_text = resumes['Resume_str'][0]  # Take the first resume as an example
job_description_text = job_descriptions['Job Description'][0]  # Take the first job description as an example

# Encode the texts
resume_embedding = model.encode(resume_text)
job_description_embedding = model.encode(job_description_text)

print("Resume Embedding:", resume_embedding)
print("Job Description Embedding:", job_description_embedding)


# In[17]:


#This indicates the size of the resume_embedding matrix (384,)
#In other words its basically a matrix of with row 384 with no columns 
print(resume_embedding.shape)


# We analyzed two resumes:
# 
# 1. **First resume**: Classified as "HR."
# 2. **Second resume**: Classified as "Designer."
# 
# These two categories are clearly quite different. To verify this, we used **cosine similarity** to compare the text from the resumes after transforming them into matrices using the SentenceTransformer model.
# 
# The cosine similarity score came out to **0.37979**, which confirms that the two resumes are quite dissimilar. This result aligns with our expectation that HR and Designer resumes should differ significantly.
# 
# This demonstrates that the SentenceTransformer model effectively captures the differences between resume texts.
# 

# In[18]:


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example: Encode the first resume and its category
hr_resume_text = resumes['Resume_str'][0]  
designer_resume_text = resumes['Resume_str'][110]  


# Encode the texts using SBERT
hr_embedding = model.encode(hr_resume_text)
designer_embedding = model.encode(designer_resume_text)

# Calculate the cosine similarity between the resume and category
similarity_score = cosine_similarity([hr_embedding], [designer_embedding])[0][0]

#print("Resume:", resume_text)
#print("Category:", category_text)
print("Similarity Score:", similarity_score)


# These two resumes were both classified under the HR category. To verify their similarity, we again calculated the cosine similarity score between the resumes using their SentenceTransformer embeddings.
# 
# Cosine similarity score: **0.84231**
# 
# A higher score (closer to 1) indicates strong similarity between the two resumes in terms of semantic content. This result aligns with the expectation that resumes from the same category, such as HR, will have similar textual structures and content.
# This demonstrates the SentenceTransformer model’s effectiveness in grouping semantically similar texts.

# In[19]:


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example: Encode the first resume and its category
hr_resume_text = resumes['Resume_str'][0]  
hr_2_resume_text = resumes['Resume_str'][1]  

hr_embedding = model.encode(resume_text)
hr_2_embedding = model.encode(hr_2_resume_text)

# Calculate the cosine similarity between the resume and category
similarity_score = cosine_similarity([hr_embedding], [hr_2_embedding])[0][0]

#print("Resume:", resume_text)
#print("Category:", category_text)
print("Similarity Score:", similarity_score)


# The code below uses TF-IDF to represent resumes as word frequency matrices and calculates cosine similarity to compare resumes within and across categories.

# In[25]:


# tfidf vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# tfidf comparison between two resume str with the same category
# both of the following resumes have the same category HR
resume_one_hr = resumes['Resume_str'][0]
resume_two_hr = resumes['Resume_str'][1]
# resume with category DESIGNER
resume_designer = resumes['Resume_str'][110]
resume_aviation = resumes['Resume_str'][2367]
resume_fitness = resumes['Resume_str'][792]

# print categories for the aboce
print(resumes['Category'][0])
print(resumes['Category'][1])
print(resumes['Category'][110])
print(resumes['Category'][2367])
print(resumes['Category'][792])

# Initialize the TfidfVectorizer
tfidf = TfidfVectorizer()

# tfidf on the resumes (This will generate a row/column matrix) (Each new row repersents a new resume, the column repersents the diffrent words)
resume_tfidf = tfidf.fit_transform([resume_one_hr, resume_two_hr])

# Calculate the cosine similarity between the two resumes ()
similarity_score = cosine_similarity(resume_tfidf)

# tfidf on the resumes
resume_different_cat_tfidf = tfidf.fit_transform([resume_one_hr, resume_designer])

# Calculate the cosine similarity between the two resumes
similarity_score_different_cat = cosine_similarity(resume_different_cat_tfidf)

# tfidf on the resumes
fitness_aviation = tfidf.fit_transform([resume_aviation, resume_fitness])

# Calculate the cosine similarity between the two resumes
similarity_score_fitavi = cosine_similarity(fitness_aviation)

# print

print("Similarity Score between similar ones:", similarity_score[0][1])
print("Similarity Score between different ones:", similarity_score_different_cat[0][1])
print("Similarity Score between fitness and aviation:", similarity_score_fitavi[0][1])


# **SenteceTransformer vs TF-IDF**
# 
# As seen above, we tested both TF-IDF and SentenceTransformer and even though the similarity score when both resumes are 'HR' is higher in TFIDF (0.684 vs 0.677), higher similarity scores in TF-IDF can sometimes arise from keyword overlaps, which may not translate to better model performance. Since the two resumes share many common words, TF-IDF gives a high score even if the resumes differ significantly in meaning.
# SentenceTransformer embeddings generalize better because they look beyond word overlap, considering context and meaning, which is particularly useful for resumes written with varying language styles. SentenceTransformer produces dense embeddings (fixed size of 384 dimensions) that preserve semantic relationships between words, making it robust for downstream tasks while TF-IDF generates sparse matrices that grow with the vocabulary size, and can be inefficient and prone to overfitting in high-dimensional datasets.
# 
# Moreover, TF-IDF can unintentionally leak information from the dataset. This happens when the vocabulary is built across the entire dataset, causing some terms to be unfairly weighted during feature extraction. This could lead to artificially inflated performance in training but poor generalization in real-world scenarios, which we saw in our first drafts of this code. 
# 
# Also, SentenceTransformer embeddings are computationally efficient and well-suited for high-dimensional data, whereas TF-IDF's performance declines as the dataset size increases.
# 
# Therefore, all in all, SentenceTransformer is better suited for this project due to its ability to handle semantic relationships, prevent data leakage, and provide computational efficiency.

# So what other features can we engineer? To increase the accuracy of the model we focused on feature engineering techniques that align with the semantic nature of our dataset:
# 
# **Resume Length**: Added a numerical feature capturing the character count of each resume to account for verbosity as a potential differentiator between categories.
# 
# **SentenceTransformer Embeddings**: Generated 384-dimensional embeddings to capture the semantic structure of each resume.
# 
# **Category-Specific Keyword Counts**: Created a 24-dimensional feature vector representing the frequency of predefined keywords for each job category. This helps link resumes to their respective industries based on keyword presence.
# 
# These features enhance the dataset by combining basic metadata, semantic meaning, and category-specific relevance. They provide a robust foundation for classification tasks.
# 
# 

# In[11]:


#Creating a predefined dictionary for all 24 job Categories 
#
resume_keywords_dict = {
    "INFORMATION-TECHNOLOGY": [
        "Python", "Java", "C++", "JavaScript", "HTML", "CSS", "SQL", "NoSQL", "React", "Angular",
        "Django", "Flask", "Git", "AWS", "Azure", "Google Cloud Platform", "DevOps", "Docker",
        "Kubernetes", "Linux", "Windows Server", "Machine Learning", "Data Science", "TensorFlow",
        "Keras", "Cybersecurity", "Penetration Testing", "Scrum", "Agile", "REST APIs",
        "Microservices", "Networking", "Cloud Computing", "Virtualization", "Data Analysis"
    ],
    "BUSINESS-DEVELOPMENT": [
        "Market Research", "Business Strategy", "Negotiation", "Sales Pipeline", "Cold Calling",
        "Lead Generation", "Partnership Development", "Customer Relationship Management (CRM)",
        "Account Management", "Revenue Growth", "KPI Analysis", "B2B Sales", "B2C Sales",
        "Competitor Analysis", "Proposal Writing", "Networking", "Forecasting", "Client Retention",
        "Brand Awareness", "Cross-functional Collaboration"
    ],
    "FINANCE": [
        "Financial Analysis", "Budgeting", "Forecasting", "Tax Preparation", "Accounting Principles",
        "Auditing", "Risk Management", "Investment Analysis", "Portfolio Management", "Equity Research",
        "Valuation Models", "Financial Statements", "SAP", "QuickBooks", "IFRS", "GAAP",
        "Cost Control", "Treasury Management", "Credit Analysis", "Derivatives", "Hedge Funds",
        "Capital Markets", "Compliance", "Fixed Income", "Financial Planning"
    ],
    "ADVOCATE": [
        "Legal Research", "Litigation", "Contract Drafting", "Case Management", "Legal Writing",
        "Compliance", "Mediation", "Arbitration", "Corporate Law", "Intellectual Property Law",
        "Employment Law", "Criminal Defense", "Court Proceedings", "Discovery Process",
        "Civil Litigation", "Client Advocacy", "Deposition", "Legal Strategy", "Trial Preparation",
        "Human Rights", "Legal Negotiation", "Case Briefing"
    ],
    "ACCOUNTANT": [
        "Financial Statements", "Auditing", "Tax Returns", "Bookkeeping", "Payroll Processing",
        "Budgeting", "Forecasting", "Accounts Payable", "Accounts Receivable", "General Ledger",
        "Cost Accounting", "Variance Analysis", "IFRS", "GAAP", "SAP", "QuickBooks",
        "Microsoft Excel", "Tax Compliance", "Internal Controls", "Bank Reconciliation",
        "Financial Reporting", "Cash Flow Management"
    ],
    "ENGINEERING": [
        "CAD Software", "SolidWorks", "MATLAB", "Finite Element Analysis (FEA)", "Thermodynamics",
        "Circuit Design", "Control Systems", "Structural Analysis", "Prototyping", "AutoCAD",
        "PLC Programming", "Fluid Mechanics", "HVAC Design", "Project Management", "Lean Manufacturing",
        "Quality Assurance", "Testing and Validation", "Renewable Energy Systems", "3D Printing",
        "Technical Drawings", "Root Cause Analysis", "Failure Analysis"
    ],
    "CHEF": [
        "Menu Planning", "Food Preparation", "Culinary Techniques", "Recipe Development",
        "Inventory Management", "Food Safety", "Sanitation Standards", "Pastry Skills",
        "Grill Station", "Sous Vide Cooking", "Knife Skills", "Garnishing", "Plating",
        "Baking", "Kitchen Operations", "Staff Management", "Cost Control",
        "Menu Costing", "Customer Service", "Catering"
    ],
    "AVIATION": [
        "Flight Operations", "Air Traffic Control", "Aircraft Maintenance", "Navigation Systems",
        "Safety Protocols", "Avionics", "Aerodynamics", "Flight Planning", "Ground Operations",
        "Pilot Training", "Flight Scheduling", "Weather Analysis", "Emergency Procedures",
        "Aircraft Systems", "Communication Systems", "FAA Regulations", "Flight Logs",
        "Cabin Safety", "Fuel Management", "Airline Operations"
    ],
    "FITNESS": [
        "Personal Training", "Exercise Programming", "Nutritional Guidance", "Strength Training",
        "Cardio Workouts", "Group Classes", "Injury Prevention", "Stretching Techniques",
        "Weight Loss Programs", "Muscle Building", "Yoga", "Pilates", "Fitness Assessments",
        "Athletic Performance", "Sports Training", "Body Composition Analysis",
        "Rehabilitation Exercises", "Motivational Coaching", "Customer Engagement",
        "Wellness Education"
    ],
    "SALES": [
        "Lead Generation", "Sales Forecasting", "Account Management", "Customer Retention",
        "Cold Calling", "Closing Sales", "CRM Software", "Upselling", "Cross-Selling",
        "Prospecting", "Negotiation", "Territory Management", "Sales Presentations",
        "Pipeline Management", "Customer Insights", "Target Achievement", "Retail Operations",
        "Direct Sales", "B2B Sales", "Sales Reporting"
    ],
    "BANKING": [
        "Banking Operations", "Customer Relationship Management", "Loan Processing", "Compliance",
        "Account Management", "Risk Assessment", "Mortgage Processing", "KYC Compliance",
        "Investment Banking", "Credit Analysis", "Treasury Management", "Retail Banking",
        "Commercial Banking", "Cash Handling", "Fraud Prevention", "Financial Regulations",
        "Sales Management", "Customer Retention"
    ],
    "HEALTHCARE": [
        "Patient Care", "Medical Records", "Clinical Assessments", "Pharmacology",
        "Surgical Assistance", "Medical Terminology", "EMR Systems", "Health Education",
        "Laboratory Testing", "Physical Therapy", "Medication Administration",
        "Diagnostic Imaging", "Infection Control", "Critical Care", "Health Management",
        "Patient Advocacy", "Mental Health", "Medical Billing", "Nutrition Counseling",
        "Emergency Response"
    ],
    "CONSULTANT": [
        "Market Research", "Process Improvement", "Stakeholder Communication",
        "Performance Analysis", "Business Strategy", "Presentation Skills",
        "Cost Optimization", "Industry Analysis", "Data-Driven Insights",
        "Implementation Plans", "Change Management", "Project Delivery", "Benchmarking",
        "Problem Solving", "Workflow Optimization", "Best Practices", "Client Relations",
        "Policy Review", "System Analysis", "Technical Documentation"
    ],
    "CONSTRUCTION": [
        "Project Management", "Blueprints", "Site Inspection", "Construction Safety",
        "Cost Estimation", "Contract Management", "Building Codes", "Civil Engineering",
        "HVAC Systems", "Electrical Wiring", "Carpentry", "Masonry", "Plumbing",
        "Heavy Equipment Operation", "Structural Analysis", "Quality Assurance",
        "Site Supervision", "Supply Chain Management", "Permitting"
    ],
    "PUBLIC-RELATIONS": [
        "Media Relations", "Press Releases", "Corporate Communications", "Event Planning",
        "Crisis Management", "Social Media Strategy", "Brand Management", "Content Creation",
        "Public Speaking", "Reputation Management", "Community Outreach", "Stakeholder Engagement",
        "Influencer Marketing", "Digital PR", "Storytelling", "Networking", "Analytics Reporting"
    ],
    "HR": [
        "Recruitment", "Onboarding", "Employee Relations", "Payroll Processing",
        "Training and Development", "Performance Management", "Benefits Administration",
        "HRIS Systems", "Compliance", "Diversity and Inclusion", "Talent Acquisition",
        "Employee Engagement", "HR Policies", "Conflict Resolution", "Succession Planning"
    ],
    "DESIGNER": [
        "Graphic Design", "UX/UI Design", "Adobe Creative Suite", "Sketch", "Figma",
        "Prototyping", "Wireframing", "Visual Design", "Typography", "Color Theory",
        "Brand Identity", "Motion Graphics", "Web Design", "Interaction Design",
        "Illustration", "Animation", "3D Modeling", "User Research", "Design Thinking"
    ],
    "ARTS": [
        "Painting", "Sculpting", "Photography", "Sketching", "Digital Art",
        "Art Curation", "Illustration", "Mixed Media", "Exhibition Planning",
        "Art Installation", "Visual Storytelling", "Creative Direction", "Printmaking",
        "Art Education", "Portfolio Development"
    ],
    "TEACHER": [
        "Lesson Planning", "Classroom Management", "Curriculum Development",
        "Student Assessment", "Differentiated Instruction", "ESL Instruction",
        "Behavior Management", "Educational Technology", "Special Education",
        "Parent Communication", "Subject Expertise", "Professional Development",
        "Tutoring", "Learning Objectives", "Student Engagement"
    ],
    "APPAREL": [
        "Fashion Design", "Textile Production", "Pattern Making", "Garment Construction",
        "Retail Merchandising", "Trend Forecasting", "Fashion Illustration", "Apparel Marketing",
        "Fabric Analysis", "Sewing Techniques", "Fashion Styling", "Clothing Alterations",
        "Production Scheduling", "Quality Control", "Brand Development"
    ],
    "DIGITAL-MEDIA": [
        "Content Creation", "SEO Optimization", "Social Media Strategy", "Email Marketing",
        "Google Analytics", "Copywriting", "Video Editing", "Graphic Design",
        "Influencer Marketing", "Pay-Per-Click Advertising", "Search Engine Marketing",
        "Social Media Management", "Web Analytics", "Content Marketing"
    ],
    "AGRICULTURE": [
        "Crop Management", "Farm Equipment", "Horticulture", "Soil Testing", "Irrigation Systems",
        "Pest Control", "Agronomy", "Livestock Management", "Sustainable Farming", "Organic Certification",
        "Farm Operations", "Yield Optimization", "Composting", "Agricultural Policy", "Crop Rotation"
    ],
    "AUTOMOBILE": [
        "Vehicle Diagnostics", "Automotive Repair", "AutoCAD for Vehicles", "Electric Vehicles",
        "Engine Tuning", "Vehicle Testing", "Automotive Engineering", "Vehicle Maintenance",
        "Fuel Systems", "Hybrid Technology", "Brake Systems", "Suspension Systems", "Car Design",
        "Vehicle Inspections", "Automobile Electronics"
    ],
    "BPO": [
        "Customer Support", "Technical Support", "Call Handling", "CRM Software", "Quality Assurance",
        "Inbound Calls", "Outbound Calls", "Data Entry", "Voice Process", "Non-Voice Process",
        "Upselling", "Cross-Selling", "Email Support", "Chat Support", "Team Management",
        "Process Optimization", "Shift Management", "Customer Retention", "Escalation Handling"
    ]
}


# **From this point on we will be feature engineering**

# Using the predefined dictionary of keywords for 24 categories, we created an array for each resume where each index represents a category, and the value indicates the frequency of keywords found for that category.
# 
# For example, when applied to five resumes, the resulting arrays showed high keyword counts at index 0, corresponding to the **Information Technology (IT)** category. The order of the categories in the array matches the order of the categories listed in the dictionary.

# In[12]:


def calculate_scores(text, keywords_dict):
    scores = []
    for category, keywords in keywords_dict.items():
        # Calculate the count of keywords in the text for each category
        score = sum(text.lower().count(keyword.lower()) for keyword in keywords)
        scores.append(score)
    return scores

# Apply the scoring function and store the result as a single column
resumes["Category_Scores"] = resumes["Resume_str"].apply(lambda x: calculate_scores(x, resume_keywords_dict))


# In[16]:


for i in range(220,225): 
    print(resumes['Category'][i])
    print(resumes['Category_Scores'][i])


# We are adding a new column called `SentenceTransform` to the dataset. This column will store the **384-dimensional matrix** generated by the SentenceTransformer model for each resume.
# 
# These embeddings capture the **semantic meaning** of the text, meaning resumes with similar content will have embeddings that are closer to each other in higher-dimensional space. This makes the embeddings useful for identifying similar resumes based on their semantic structure and meaning.

# In[13]:


from sentence_transformers import SentenceTransformer
import pandas as pd
from tqdm import tqdm


# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

#We use tqdm to see a progress bar as we apply (This column generation will take some time)
tqdm.pandas()

# Apply SentenceTransformer with a progress bar
resumes['SentenceTransform'] = resumes['Resume_str'].progress_apply(lambda x: model.encode(x))


# In[14]:


print(resumes.columns)


# **Training the SVM model ()**

# In[14]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import numpy as np


# Encode the category labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(resumes['Category']) #Y is the encoded category (Numerically repersented)


sentence_transform_matrix = np.vstack(resumes['SentenceTransform'].values) # Shape will be (n,384) 
category_scores_matrix = np.array(resumes['Category_Scores'].tolist())  #Shape will be (n,24) number of categoires 
resume_length = resumes['resume_length'].values.reshape(-1, 1) #Shape will be (n,1) where n is number of resumes

#We will horizontally stack the the diffrent categories into one array (See explanation above)
X = np.hstack([resume_length, category_scores_matrix, sentence_transform_matrix])





# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)


# **The following is the training / classfication report for the LogisticRegression**
# 

# In[23]:


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, penalty='l2', solver='lbfgs')
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# Print the classification report
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


# **The below is the Basic RandomForestClassifier Model Performance
# Refer to the Confusion Matrix**

# In[22]:


from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest classifier with verbose output
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, verbose=1)

# Train the classifier
rf_classifier.fit(X_train, y_train)

from sklearn.metrics import classification_report

# Predict the labels for the test set
y_pred = rf_classifier.predict(X_test)

# Generate the classification report
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

# Display the report
print("Classification Report:")
print(report)


# We have tested LogisticRegression, RandomForest, SVC (both rbf,linear kernel), and XGBoost.
# 
# We only show the LogisticRegression and RandomForest as both SVC and XGBoost models were to computationally expensive to run (further details on the Readme).
# 
# Thus, we decided to stick with the basic LogisticRegression model and Hyptertune

# ****The following below is the hyptertunning of the LogisticRegression Model****

# In[28]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000)

# Define hyperparameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],              # Regularization strength
    'penalty': ['l2'],                         # Regularization type
    'solver': ['lbfgs', 'saga'],               # Solvers
    'class_weight': [None, 'balanced']         # Handle class imbalance
}

# Perform GridSearchCV
grid_search = GridSearchCV(
    estimator=logistic_model,
    param_grid=param_grid,
    scoring='f1_weighted',  # Weighted F1-score to handle imbalance
    cv=5,                   # 5-fold cross-validation
    verbose=3,              # Highest verbosity to track progress
    n_jobs=-1               # Use all available CPU cores
)

# Fit the GridSearchCV model
grid_search.fit(X_train_scaled, y_train)

# Print best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Weighted F1 Score:", grid_search.best_score_)

# Predict on the test set using the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

# Evaluate the best model on the test set
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))





# After performing hyperparameter tuning using GridSearchCV, the following parameters were identified as optimal:
# 
# **C=0.01**: A low regularization strength to prevent overfitting on high-dimensional data.
# 
# **class_weight='balanced'**: Ensures that minority classes receive higher importance during training, improving F1-scores for underrepresented categories.
# 
# **penalty='l2'**: Ridge regularization to prevent multicollinearity in feature sets.
# 
# **solver='saga'**: Efficient for large datasets with sparse features.
# 
# 
# Resulting Performance:
# 
# Weighted F1-Score: **0.7039** 
# 
# Improvement Over Baseline: Significant improvement, particularly for minority classes like AGRICULTURE and AUTOMOBILE.
# 
# These parameters optimize the model’s ability to handle class imbalance and high-dimensional embeddings effectively.
# 
# 
# 
# 
# 
# 

# 

# In[31]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize Logistic Regression with the best parameters
logistic_model = LogisticRegression(
    C=0.01, 
    class_weight='balanced', 
    penalty='l2', 
    solver='saga', 
    max_iter=1000
)

# Train the model
logistic_model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = logistic_model.predict(X_test_scaled)

# Generate and print classification report
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()






# Fourth Visualization Above: **Confusion Matrix**

# 
