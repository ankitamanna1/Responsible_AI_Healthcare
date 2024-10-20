
**Responsible AI in Healthcare Recommendation System(Hospital readmission due to diabetes**

This repository contains the code and methodologies used for my dissertation, where I explored the application of Responsible AI in a healthcare recommendation system. I specifically focused on identifying and mitigating bias using AI Fairness 360 (AIF360) to ensure fairness across different demographic groups. The project leverages machine learning models to make recommendations while maintaining fairness and accuracy.

- Table of Contents
- Project Overview
- Motivation
- Methodology
- Data Preprocessing
- Modeling
- Bias Detection and Mitigation
- Results
- Technologies Used

**Project Overview******
The main goal of this project is to ensure fairness and transparency in a healthcare recommendation system by identifying and mitigating bias during the machine learning model development process. Bias in healthcare recommendations can have serious consequences, so fairness metrics such as disparate impact were used to detect any unfair outcomes. Bias was mitigated using both preprocessing and in-processing algorithms from the AIF360 library.

**Motivation**
As AI-driven systems become more integrated into healthcare, there is an increasing need to ensure that these systems are fair and do not unintentionally discriminate against particular demographic groups. This project aims to detect, analyze, and mitigate bias in a healthcare recommendation system, contributing to responsible AI development.

**Methodology**
Data Preprocessing
Data Cleaning: The dataset was thoroughly cleaned, including handling missing values and outliers.
Feature Engineering: Additional features were created based on domain knowledge to improve model performance.
Modeling
**Five machine learning models were trained on the dataset:**

Logistic Regression
Random Forest
Support Vector Machine (SVM)
Decision Tree
Logistic Regression
XGBoost (Best in terms of accuracy)
XGBoost was found to be the best-performing model with the highest accuracy. Hyperparameter tuning was conducted to optimize its performance.

**Bias Detection and Mitigation**
Bias Detection (Initial Stage):
Bias was evaluated using fairness metrics such as disparate impact from the AIF360 library. At the data preprocessing stage, there was no significant bias detected in the raw dataset.

**Bias Detection (Post-Modeling):**
After applying the machine learning models, an increase in bias was detected in the predictions. This was measured by fairness metrics such as disparate impact.

**Bias Mitigation:**
Preprocessing Algorithms: Techniques such as re-weighting and data transformation were applied to mitigate bias before model training.
In-Processing Algorithms: Models were modified during the training process using techniques like adversarial debiasing and prejudice remover to reduce the bias.
After applying these algorithms, the bias was successfully mitigated without significant loss in model performance.
**Results**
Best Performing Model: XGBoost achieved the highest accuracy after hyperparameter tuning.
Bias Mitigation: While bias was initially introduced after model training, both preprocessing and in-processing methods successfully mitigated bias, ensuring fairness in the final recommendation system.
Technologies Used
Python
AIF360 (AI Fairness 360) for bias detection and mitigation
XGBoost for modeling
scikit-learn for model training and evaluation
pandas, numpy for data manipulation
matplotlib, seaborn for data visualization
