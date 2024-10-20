#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import time
import timeit
# execution time
from timeit import default_timer as timer
from datetime import timedelta


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, log_loss, recall_score 
from sklearn.metrics import precision_recall_fscore_support
from timeit import default_timer as timer
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import imblearn
from imblearn.over_sampling import RandomOverSampler, SMOTE


# In[2]:


from aif360.datasets import BinaryLabelDataset #dataset 
from aif360.explainers import MetricTextExplainer # Explainer
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric # Metrics
from aif360.algorithms.preprocessing.reweighing import Reweighing

from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.algorithms.inprocessing import AdversarialDebiasing

from aif360.explainers import Explainer
from aif360.metrics import Metric



# In[3]:


raw_df = pd.read_csv('diabetic_dataset.csv')
raw_df.info()


# In[4]:


raw_df.shape


# In[5]:


raw_df['race'] = raw_df['race'].replace('?', 'Other')


# In[6]:


raw_df['race'].unique()


# In[7]:


raw_df.drop('medical_specialty',axis=1)


# In[8]:


# Remove rows where 'admission_type' is 'NA' or 'NULL' (5=NA,6=NULL)
raw_df = raw_df[~raw_df['admission_type_id'].isin([5, 6])]

# Verify the removal
print(raw_df['admission_type_id'].unique())


# In[9]:


# Remove rows where 'discharge_disposition_id' is expired(11,19,20,21),Null (18),26(unknown),25(not mapble)

raw_df = raw_df[~raw_df['discharge_disposition_id'].isin([11,19,20,21])]

# Verify the removal
print(raw_df['discharge_disposition_id'].unique())


# In[10]:


# Remove rows where 'admission_source_id' is 'UNKNOWN' or 'NULL' (17,20,21)


# In[11]:


raw_df = raw_df[~raw_df['admission_source_id'].isin([17,20,21])]
print(raw_df['admission_source_id'].unique())


# In[12]:


#Feature Engineering


# In[13]:


raw_df['No_of_total_visits'] = raw_df['number_outpatient'] + raw_df['number_outpatient']
raw_df


# In[14]:


def catg_icd9(icd_code):
    # Remove non-numeric characters for comparison purposes
    numeric_code = ''.join(filter(str.isdigit, icd_code.split('.')[0]))
    
    if numeric_code.isdigit():
        numeric_code = int(numeric_code)
        if 390 <= numeric_code <= 459 or numeric_code == 785:
            return 'Circulatory'
        elif 460 <= numeric_code <= 519 or numeric_code == 786:
            return 'Respiratory'
        elif 520 <= numeric_code <= 579 or numeric_code == 787:
            return 'Digestive'
        elif 250 <= numeric_code <= 250:
            return 'Diabetes'
        elif 800 <= numeric_code <= 999:
            return 'Injury'
        elif 710 <= numeric_code <= 739:
            return 'Musculoskeletal'
        elif 580 <= numeric_code <= 629:
            return 'Genitourinary'
        elif 140 <= numeric_code <= 239:
            return 'Neoplasms'
        else:
            return 'Other'
    elif icd_code.startswith('V'):
        return 'Other'  # V-codes are supplementary classification
    elif icd_code.startswith('E'):
        return 'Other'  # E-codes are external causes of injury
    else:
        return 'Other'


# In[15]:


raw_df['cat1']=raw_df['diag_1'].apply(catg_icd9)
raw_df['cat2']=raw_df['diag_2'].apply(catg_icd9)
raw_df['cat3']=raw_df['diag_3'].apply(catg_icd9)


# In[16]:


raw_df['cat1'].unique()


# In[17]:


def get_one_diagnosis(raw_df):
    """
    Process diagnoses to keep only one per patient based on specified rules.
    """
    # Create a new column 'diagnosis' initially set to 'diag-1'
    raw_df['diagnosis'] = raw_df['diag_1']
    
    # Update 'diagnosis' where 'cat2' equals 'cat3'
    raw_df['diagnosis'] = np.where(
        (raw_df['cat2'] == raw_df['cat3']) & (raw_df['cat2'].notnull()),
        raw_df['diag_2'],
        raw_df['diag_1']
    )
    
    return raw_df


# In[18]:


# Apply the function
result_df = get_one_diagnosis(raw_df)

# Display the resulting DataFrame
print(result_df)


# In[19]:


result_df['diagnosis']


# In[20]:


result_df['diagnosis'].apply(catg_icd9)


# In[21]:


# List of medicine columns
medicine_columns = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
    'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
    'miglitol', 'troglitazone', 'tolazamide', 'examide', 
    'citoglipton', 'insulin', 'glyburide-metformin', 
    'glipizide-metformin', 'glimepiride-pioglitazone','metformin-rosiglitazone',
    'metformin-pioglitazone'
]

# Create the new column 'dosage_up_down' initialized to 0
result_df['dosage_up_down'] = 0

# Update 'dosage_up_down' based on the dosage change in each medicine
for col in medicine_columns:
    result_df['dosage_up_down'] += result_df[col].apply(lambda x: 1 if x == 'Up' else -1 if x == 'Down' else 0)

# Check the result
print(result_df[['dosage_up_down']])


# In[22]:


result_df['dosage_up_down'].unique()


# In[23]:


result_df['A1Cresult'].unique()


# In[24]:


result_df['A1_converted']=result_df['A1Cresult'].apply(lambda x: 1 if x in ['>8', '>7'] else 0)


# In[25]:


result_df['A1_converted'].unique()


# In[26]:


result_df['max_glu_serum'].unique()


# In[27]:


def convert_glu_serum(value):
    if pd.isna(value):  # Check for NaN values
        return 0
    elif value == '>300':
        return 3
    elif value == '>200':
        return 2
    elif value == 'Norm':
        return 0
    else:
        return value 


# In[28]:


result_df['max_glu_serum'] = result_df['max_glu_serum'].apply(convert_glu_serum)


# In[29]:


result_df['max_glu_serum'].unique()


# In[30]:


# Filter the DataFrame to include only rows where 'readmission' is '<30' or 'no'(readmission <30 days,and no)
#result_df = result_df[result_df['readmitted'].isin(['<30', 'NO'])]


# In[31]:


# Replace '<30' with 1 and 'no' with 0
# Replace '<30' with 1 and 'no' with 0
#result_df['readmitted'] = result_df['readmitted'].replace({'<30': 1, 'NO': 0})


# In[32]:


result_df = result_df[result_df['readmitted'].isin(['<30', 'NO'])]


# In[33]:


result_df['readmitted']=result_df['readmitted'].apply(lambda x: 1 if x in ['<30','>30'] else 0)


# In[34]:


result_df['readmitted'].unique()


# In[35]:


result_df['readmitted'].unique()


# In[ ]:





# In[36]:


result_df['admission_type_id']


# In[37]:


# Group by the 'age_group' column and count the number of entries in each group
age_group_counts = result_df['age'].value_counts().sort_index()
age_group_counts


# In[38]:


result_df['age'] = result_df['age'].str.replace('[', '', regex=False).str.replace(')', '', regex=False)


# In[39]:


age_group_counts = result_df['age'].value_counts().sort_index()
age_group_counts


# In[40]:


# Split the age ranges into two separate columns
result_df[['age_start', 'age_end']] = result_df['age'].str.split('-', expand=True)

# Convert the new columns to numeric
result_df['age_start'] = pd.to_numeric(result_df['age_start'], errors='coerce')
result_df['age_end'] = pd.to_numeric(result_df['age_end'], errors='coerce')

# Display the DataFrame after splitting and converting
print("\nDataFrame after splitting the ranges and converting to numeric:")
print(raw_df)


# In[41]:


bins = [0, 30, 60, 100]
labels = ['0-30', '30-60', '60-100']

# Function to categorize age ranges into the defined bins
def categorize_age_range(row):
    if row['age_start'] >= 0 and row['age_end'] <= 30:
        return '0-30'
    elif row['age_start'] >= 30 and row['age_end'] <= 60:
        return '30-60'
    elif row['age_start'] >= 60 and row['age_end'] <= 100:
        return '60-100'
    else:
        return 'Other'

# Apply the function to create a new 'age_bin' column
result_df['age_bin'] = result_df.apply(categorize_age_range, axis=1)


# In[42]:


result_df.columns


# In[43]:


result_df = result_df.drop(columns=['diag_1', 'diag_2', 'diag_3','age', 'number_inpatient','number_outpatient','metformin',
       'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
       'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
       'tolazamide', 'examide', 'citoglipton', 'insulin',
       'glyburide-metformin', 'glipizide-metformin',
       'glimepiride-pioglitazone', 'metformin-rosiglitazone',
       'metformin-pioglitazone', 'A1Cresult','age_start','age_end'])


# In[44]:


result_df.columns


# In[45]:


raw_df=result_df.copy()


# In[46]:


#result_df = result_df.drop(columns=['max_glu_serum','gender','admission_type_id','discharge_disposition_id','admission_source_id','medical_specialty','diabetesMed','diagnosis','A1_converted','age_bin','change'])


# In[47]:


#Target -'readmitted'


# In[48]:


raw_df['readmitted'].unique()


# In[49]:


plt.figure(figsize=(3,4))
raw_df['readmitted'].value_counts().plot(kind='bar', rot=0, color='#6495ED')
plt.title('readmitted')
plt.show()


# In[50]:


num_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 
            'number_diagnoses']
raw_df[num_cols].describe()


# In[51]:


num_plots = len(num_cols)
cols_per_row = 3
rows = (num_plots - 1) // cols_per_row + 1

fig, axes = plt.subplots(nrows=rows, ncols=cols_per_row, figsize=(16, 4))
axes = axes.flatten()
sns.set_style("white")

for i, column in enumerate(num_cols):
    ax = axes[i]
    sns.boxplot(data=raw_df, x=column, ax=ax, linewidth=0.5,
                color='#2DB2C4', width=0.5)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(column)

for i in range(num_plots, len(axes)):
    fig.delaxes(axes[i])
    
plt.tight_layout()
plt.show()


# In[52]:


num_plots = len(num_cols)
cols_per_row = 3
rows = (num_plots - 1) // cols_per_row + 1

fig, axes = plt.subplots(nrows=rows, ncols=cols_per_row, figsize=(16, 8))
axes = axes.flatten()
sns.set_style("white")

for i, column in enumerate(num_cols):
    ax = axes[i]
    sns.histplot(data=raw_df, x=column, ax=ax,
                 palette=['#2DB2C4','#C42D2D'],
                 bins=10, multiple='stack')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(column)

for i in range(num_plots, len(axes)):
    fig.delaxes(axes[i])
    
plt.tight_layout()
plt.show()


# In[53]:


# Plotting the histogram
plt.hist(raw_df['num_medications'], bins=10, edgecolor='black')

# Adding title and labels
plt.title('num_medications')
plt.xlabel('Value')
plt.ylabel('Number')

# Display the plot
plt.show()


# In[54]:


# Plotting the histogram
plt.hist(raw_df['num_procedures'], bins=7, edgecolor='black')

# Adding title and labels
plt.title('num_procedures')
plt.xlabel('Value')
plt.ylabel('Number')

# Display the plot
plt.show()


# In[55]:


#adjustment for Schweness of histogram


# In[56]:


raw_df['time_in_hospital'] = np.sqrt(raw_df['time_in_hospital'])
raw_df['No_of_total_visits'] = np.sqrt(raw_df['No_of_total_visits'])
raw_df['number_emergency'] = np.sqrt(raw_df['number_emergency'])
raw_df['num_procedures'] = np.sqrt(raw_df['num_procedures'])
raw_df['num_medications'] = np.sqrt(raw_df['num_medications'])





# In[57]:


plt.hist(raw_df['num_procedures'], bins=4, edgecolor='black')

# Adding title and labels
plt.title('num_procedures')
plt.xlabel('Value')
plt.ylabel('Number')

# Display the plot
plt.show()


# In[58]:


plt.hist(raw_df['time_in_hospital'], bins=4, edgecolor='black')

# Adding title and labels
plt.title('time_in_hospital')
plt.xlabel('Value')
plt.ylabel('Number')

# Display the plot
plt.show()


# In[59]:


#Categorical columns


# In[60]:


raw_df.columns


# In[61]:


# Assuming raw_df is already defined and contains the required data
cat_col = ['race', 'gender', 'diabetesMed', 'age_bin']
cat_plots = len(cat_col)
cols_per_row = 4
rows = (cat_plots + cols_per_row - 1) // cols_per_row

fig, axes = plt.subplots(nrows=rows, ncols=cols_per_row, figsize=(12, 3 * rows))
axes = axes.flatten()
sns.set_style("white")

for i, column in enumerate(cat_col):
    ax = axes[i]
    # Create a contingency table
    contingency_tab = pd.crosstab(raw_df[column], raw_df['readmitted'])
    # Plot using seaborn barplot for each level of the categorical variable
    contingency_tab.plot(kind='bar', ax=ax, color=['#1f77b4', '#87CEEB'])
    ax.set_xlabel(column)
    ax.set_ylabel('Count')
    ax.set_title(column)
    ax.legend(title='readmitted')

# Remove empty subplots if the number of plots doesn't fill the entire grid
for i in range(cat_plots, len(axes)):
    fig.delaxes(axes[i])

# Adjust layout
plt.tight_layout()
plt.show()


# In[62]:


num_cols_new = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 
            'number_diagnoses','number_emergency','No_of_total_visits']


# In[63]:


raw_df.columns


# In[64]:


# Calculate the correlation matrix
corr_matrix = raw_df[num_cols_new].corr()
corr_matrix


# In[65]:


# Custom blue colormap
blue_cmap = sns.diverging_palette(240, 240, s=100, l=40, n=9, as_cmap=True)

# Create the heatmap
plt.figure(figsize=(8, 8))
sns.heatmap(data=corr_matrix, vmin=-1, vmax=1, cmap=blue_cmap, annot=True, cbar=True, fmt=".2f", linewidths=0.1)
plt.title('Correlation Matrix')
plt.show()


# In[66]:


import seaborn as sns
import matplotlib
print("Seaborn version:", sns.__version__)
print("Matplotlib version:", matplotlib.__version__)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[67]:


plt.figure(figsize=(6, 6))
ax = sns.heatmap(
    data=corr_matrix, 
    vmin=-1, vmax=1, 
    cmap='YlGnBu', 
    annot=True,       # Ensure annotations are enabled
    cbar=True, 
    fmt=".2f",        # Format for annotation
    linewidths=0.1    # Width of lines between cells
)

# Title for the heatmap
plt.title('Correlation Matrix with Box Labels')
plt.show()


# In[68]:


plt.figure(figsize=(6,4))
ax = sns.heatmap(
    data=corr_matrix, 
    vmin=-1, vmax=1, 
    cmap='YlGnBu', 
    annot=True,       # Ensure annotations are enabled
    cbar=True, 
    fmt=".2f",        # Format for annotation
    linewidths=0.1    # Width of lines between cells
)

# Title for the heatmap
plt.title('Correlation Matrix with Box Labels')
plt.show()


# In[69]:


# Define the mapping for races
race_mapping = {
    'Caucasian': 1,
    'AfricanAmerican': 0,
    'Asian': 0,
    'Hispanic': 0,
    'Other': 0  # Assuming 'other' is another category in your data, if not you can include other specific races
}

# Map the race column to numerical values
df=raw_df.copy()
raw_df['race'] = raw_df['race'].replace(race_mapping)


# In[70]:


raw_df['race'].unique()


# In[71]:


raw_df.info()


# In[72]:


# Manually specify the columns to be converted
categorical_columns = ['race','admission_type_id','admission_source_id','medical_specialty','max_glu_serum','readmitted','A1_converted']

# Convert specified columns to 'category' dtype
for col in categorical_columns:
    raw_df[col] = raw_df[col].astype('object')

# Verify the conversion
print(raw_df.dtypes)


# In[73]:


from scipy.stats import chi2_contingency, pointbiserialr, f_oneway


# In[74]:


# Chi-Square Test for Categorical vs Categorical
results = {}
for col in raw_df.select_dtypes(include=['object']).columns:
    if col != 'readmitted':
        contingency_table = pd.crosstab(raw_df['readmitted'], raw_df[col])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        results[col] = {'Test': 'Chi-Square', 'p-value': p}
 # Display results
results_df = pd.DataFrame(results).T
print(results_df)


# In[75]:


# ANOVA for Numerical vs Categorical
results={}
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    groups = [raw_df[col][raw_df['readmitted'] == category] for category in raw_df['readmitted'].unique()]
    f_value, p_value = f_oneway(*groups)
    results[col] = {'Test': 'ANOVA', 'p-value': p_value}
    # Display results
results_df = pd.DataFrame(results).T
print(results_df)


# In[76]:


#Not removing age as in medical data , age ia a important factor, not removing outliers as these might be sensitive for medical records.


# In[77]:


#Label encoding in categorical variables


# In[78]:


# Instantiate the LabelEncoder
from sklearn.preprocessing import LabelEncoder 
label_encoder = LabelEncoder()

# Apply label encoding to each categorical column
for col in ['gender','admission_type_id','discharge_disposition_id','admission_source_id','medical_specialty','diabetesMed','diagnosis','A1_converted','age_bin','change','max_glu_serum']:
    raw_df[col + '_label_encoded'] = label_encoder.fit_transform(raw_df[col])

raw_df.columns


# In[79]:


raw_df = raw_df.drop(columns=['max_glu_serum','gender','admission_type_id','discharge_disposition_id','admission_source_id','medical_specialty','diabetesMed','diagnosis','A1_converted','age_bin','change','encounter_id','patient_nbr','cat1','cat2','cat3']) 


# ### Main Dataset disperate impact

# In[80]:


raw_df['readmitted'].unique()


# In[81]:


raw_df['readmitted'] = raw_df['readmitted'].astype(int)


# In[82]:


raw_df['readmitted']


# In[83]:


raw_df.columns


# In[84]:


raw_df.shape


# In[85]:


# Making our data into a Binary label dataset
aif_fair = BinaryLabelDataset(favorable_label = 1, unfavorable_label = 0, df=raw_df,
                                                      label_names=["readmitted"],
                                                     protected_attribute_names=["race"],
                                              privileged_protected_attributes = [1,1])


# In[86]:


#Disparate impact measurement for Race in main dataset
#Taking the prieveleged group as Caucasian



metric_aif= BinaryLabelDatasetMetric(aif_fair,
                                      unprivileged_groups=[{"race":0}],
                                      privileged_groups=[{"race":1}]) 
explainer_aif = MetricTextExplainer(metric_aif)

print(explainer_aif.disparate_impact())


# In[ ]:





# In[ ]:





# In[87]:


train_set, test_set = train_test_split(raw_df, test_size=0.2, random_state=42)



# In[88]:


train_set_latest=train_set.copy()
test_set_latest=test_set.copy()


# Checking of readmission on trainset

# In[89]:


plt.figure(figsize=(3,4))
train_set['readmitted'].value_counts().plot(kind='bar', rot=0, color='#6495ED')
plt.title('readmitted')
plt.show()


# In[90]:


readmission_counts=raw_df['readmitted'].value_counts()


# In[91]:


readmission_percentages = (readmission_counts / len(raw_df)
) * 100
readmission_percentages


# In[92]:


x_train = train_set.drop(columns=['readmitted'])
y_train = train_set['readmitted']


# In[93]:


x_test=test_set.drop(columns=['readmitted'])
y_test=test_set['readmitted']


# In[94]:


x_train.shape


# In[95]:


y_train_encoded = label_encoder.fit_transform(y_train)


# In[96]:


y_test_encoded = label_encoder.fit_transform(y_test)


# In[97]:


#Apply smote for class imbalance


# In[98]:


# Plot class distribution before SMOTE
from collections import Counter
def plot_class_distribution(y, title):
    counter = Counter(y)
    plt.bar(counter.keys(), counter.values())
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.xticks([0, 1])
    plt.show()


# In[99]:


plot_class_distribution(y_train_encoded, 'Class Distribution Before SMOTE')


# In[ ]:





# In[100]:


# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(x_train, y_train_encoded )


# In[101]:


plot_class_distribution(y_train_res, 'Class Distribution Before SMOTE')


# In[102]:


scaler = MinMaxScaler()

# Fit the scaler on the training features and transform both train and test features
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_res), columns=x_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)


# Train the model (Base model)

# In[103]:


#Computation of confusion matrix to visualise the model predictions against the actual labels
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
def conf_mat(y_pred, ytest):
    class_names = [0,1]
    tick_marks_y = [0.5, 2]
    tick_marks_x = [0.5, 2]
    confusion_matrix = metrics.confusion_matrix(ytest, y_pred)
    confusion_matrix_df = pd.DataFrame(confusion_matrix, range(2), range(2))
    plt.figure(figsize = (6, 4.75))
    sns.set(font_scale = 1.4) # label size
    plt.title("Confusion Matrix")
    sns.heatmap(confusion_matrix_df, annot = True, annot_kws = {"size": 16}, fmt = 'd',cmap='seismic')
 # font size
    plt.yticks(tick_marks_y, class_names, rotation = 'vertical')
    plt.xticks(tick_marks_x, class_names, rotation = 'horizontal')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.grid(False)
    plt.show()


# In[104]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


# In[105]:


# Initialize the Decision Tree classifier
start = time.time()
dt_classifier = DecisionTreeClassifier(random_state=42)

# Fit the model on the training data
dt_classifier.fit(X_train_scaled, y_train_res)

# Predict on the test data
y_pred = dt_classifier.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test_encoded, y_pred)
#Time
dec_tree_duration = time.time()-start
#conf_matrix = conf_mat(y_test_encoded, y_pred)
class_report = classification_report(y_test_encoded, y_pred)
confusion_matrix = metrics.confusion_matrix(y_test_encoded, y_pred)

print(f'Execution time of Decision Tree is: {dec_tree_duration:.3f} seconds')
print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(confusion_matrix)
print('Classification Report:')
print(class_report)

conf_mat(y_test_encoded, y_pred)


# In[106]:


###Random forest Model


# In[107]:


# fit Random Forest model 
from sklearn.ensemble import RandomForestClassifier
start = time.time()
#Running Random Forest Model on the original data without implementing reweighing
model_rf = RandomForestClassifier(n_estimators=300,min_samples_split=10, max_depth=10,n_jobs=-1)
model_rf.fit(X_train_scaled, y_train_res)


# In[108]:


y_pred_test = model_rf.predict(X_test_scaled)

#Time
RF_duration = time.time()-start
acc_test = accuracy_score(y_true=y_test_encoded, y_pred=y_pred_test)
class_report_rf = classification_report(y_test_encoded, y_pred_test)

print(f'Execution time of Decision Tree is: {RF_duration:.3f} seconds')
print(f'Accuracy: {acc_test:.2f}')
print('Classification Report:')
print(class_report_rf)

conf_mat(y_test_encoded, y_pred_test)


# In[109]:


importances = model_rf.feature_importances_

# make importance relative to the max importance
feature_importance = 100.0 * (importances / importances.max())
sorted_idx = np.argsort(feature_importance)
feature_names = list(X_train_scaled.columns.values)
feature_names_sort = [feature_names[indice] for indice in sorted_idx]
pos = np.arange(sorted_idx.shape[0]) + .5


# plot the result
plt.figure(figsize=(12, 10))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, feature_names_sort)
plt.title('Relative Feature Importance', fontsize=20)
plt.show()


# In[110]:


#XGBOOST model


# In[111]:


# Initialize the XGBoost classifier
from xgboost import XGBClassifier
start = time.time()

xgb = XGBClassifier(random_state = 42)
xgb.fit(X_train_scaled, y_train_res)
y_pred_xgb = xgb.predict(X_test_scaled)



#Time
XGb_duration = time.time()-start

# Evaluate the model
accuracy_xgb = accuracy_score(y_test_encoded, y_pred_xgb)
#conf_matrix_xgb = confusion_matrix(y_test_encoded, y_pred_xgb)
class_report_xgb = classification_report(y_test_encoded, y_pred_xgb)

#Print
print(f'Execution time of XGBOOST is: {XGb_duration:.3f} seconds')
print(f'XGBoost Accuracy: {accuracy_xgb:.2f}')

print('XGBoost Classification Report:')
print(class_report_xgb)
conf_mat(y_test_encoded, y_pred_test)


# In[ ]:





# In[112]:


from sklearn.ensemble import AdaBoostClassifier


# In[113]:


# Initialize the AdaBoost classifier
ada_classifier = AdaBoostClassifier()

# Fit the model on the training data
ada_classifier.fit(X_train_scaled, y_train_res)

# Predict on the test data
y_pred_ada = ada_classifier.predict(X_test_scaled)


# In[114]:


# Evaluate the model
accuracy_ada = accuracy_score(y_test_encoded, y_pred_ada)
#conf_matrix_ada = confusion_matrix(y_test_encoded, y_pred_ada)
class_report_ada = classification_report(y_test_encoded, y_pred_ada)

#Time
Adaboost_duration = time.time()-start

print(f'Execution time of Decision Tree is: {Adaboost_duration:.3f} seconds')
print(f'AdaBoost Accuracy: {accuracy_ada:.2f}')
print('AdaBoost Classification Report:')
print(class_report_ada)
conf_mat(y_test_encoded, y_pred_ada)


# In[115]:


#Instantiate the logistic regression model
log_reg = LogisticRegression()

# Train the model on the training data
log_reg.fit(X_train_scaled, y_train_res)

# Predict the target values for the test set
y_pred_log_reg = log_reg.predict(X_test_scaled)


#Time
LR_duration = time.time()-start


# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test_encoded, y_pred_log_reg)
print(f'Accuracy: {accuracy:.2f}')

# Generate the classification report
print("\nClassification Report:")
print(classification_report(y_test_encoded, y_pred_log_reg))

# Generate the confusion matrix
print("\nConfusion Matrix:")
print(conf_mat(y_test_encoded, y_pred_log_reg))
print(f'Execution time of Decision Tree is: {LR_duration:.3f} seconds')


# In[116]:


# Instantiate the SVM classifier
from sklearn.svm import SVC
svm_model = SVC(kernel='linear', random_state=42)

# Train the model on the training data
svm_model.fit(X_train_scaled, y_train_res)
# Predict the target values for the test set
y_pred_svc = svm_model.predict(X_test_scaled)


# In[117]:


#Evaluate the accuracy of the model
accuracy = accuracy_score(y_test_encoded, y_pred_svc)
print(f'Accuracy: {accuracy:.2f}')


#Time
svm_duration = time.time()-start

# Generate the classification report
print("\nClassification Report:")
print(classification_report(y_test_encoded, y_pred_svc))
print(f'Execution time of Decision Tree is: {svm_duration:.3f} seconds')
# Generate the confusion matrix
print("\nConfusion Matrix:")
print(conf_mat(y_test_encoded, y_pred_svc))


# In[ ]:





# In[118]:


#Hyper parameter Tuning 


# #Xgboost and  Ada boost are giving best result interms of  accuracy and seconds, So I will apply hyperparameter tuning.

# In[119]:


#XGB hyperparameter tuning


# In[120]:


param_grid = {
    'eta': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'n_estimators': [100, 200, 300],
    'alpha': [0, 0.1, 1],
    'lambda': [1, 1.5, 2],
}


# In[121]:


from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(estimator=xgb, param_distributions=param_grid, 
                                   n_iter=100, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
random_search.fit(X_train_scaled, y_train_res)
xgb_duration = time.time()-start
best_params = random_search.best_params_
best_score = random_search.best_score_
xgb_duration


# In[122]:


best_params


# In[123]:


xgb_best = XGBClassifier(**best_params)
xgb_best.fit(X_train_scaled, y_train_res)

y_pred_xgboost = xgb_best.predict(X_test_scaled)
accuracy = accuracy_score(y_test_encoded, y_pred_xgboost)
print("Best Accuracy: {:.2f}%".format(accuracy * 100))


# ### Checking disperate impact after applying model 

# In[124]:


dataset_pred = test_set.copy()
dataset_pred['readmitted'] = y_pred_xgboost



# In[125]:


# Making our data into a Binary label dataset for prediction dataset
aif_fair_model = BinaryLabelDataset(favorable_label = 1, unfavorable_label = 0, df=dataset_pred,
                                                      label_names=["readmitted"],
                                                     protected_attribute_names=["race"],
                                              privileged_protected_attributes = [1,1])


# In[126]:


bin_metric = BinaryLabelDatasetMetric(aif_fair_model, 
                                     unprivileged_groups=[{"race":0}],
                                      privileged_groups=[{"race":1}]) 

disparate_impact_bin = bin_metric.disparate_impact()
print('Disparate impact: ', disparate_impact_bin)
mean_difference = bin_metric.mean_difference()
print('Mean difference: ', mean_difference)



# In[127]:


### The disperate impact of the model is showing significant bias towards privileged_groups.


# In[128]:


#Making the trainset and testset binary


# In[129]:


trainset_binary = BinaryLabelDataset(favorable_label = 1, unfavorable_label = 0, df=train_set,
                                                      label_names=["readmitted"],
                                                     protected_attribute_names=["race"],
                                              privileged_protected_attributes = [1,1])


# In[130]:


test_set_binary = BinaryLabelDataset(favorable_label = 1, unfavorable_label = 0, df=test_set,
                                                      label_names=["readmitted"],
                                                     protected_attribute_names=["race"],
                                              privileged_protected_attributes = [1,1])


# In[131]:


classif_metric = ClassificationMetric(test_set_binary,aif_fair_model, 
                                      unprivileged_groups=[{"race":0}],
                                      privileged_groups=[{"race":1}])

#classif_disparate_impact = classif_metric.disparate_impact()
avg_odds = classif_metric.average_odds_difference()
print('Average odds difference:', avg_odds)
equal_opport = classif_metric.equal_opportunity_difference()
print('Equality of opportunity:', equal_opport)
false_discovery_rate = classif_metric.false_discovery_rate_difference()
print('False discovery rate difference:', false_discovery_rate)


# In[132]:


trainset_binary_original=trainset_binary.copy()
test_set_binary_original=test_set_binary.copy()


# In[133]:


### Mitigation method(Pre-Processing)Attempt 1. Applying reweighing method on  main trainset


# In[134]:


# Making our data into a Binary label dataset for prediction dataset

#2nd cell content
RW = Reweighing(unprivileged_groups=[{"race":0}],
                privileged_groups=[{"race":1}]) 
RW.fit(trainset_binary)
dataset_tf_train = RW.transform(trainset_binary)

dataset_tf_train


# In[135]:


#3rd cell content
metric_transf = BinaryLabelDatasetMetric(dataset_tf_train,
                                          unprivileged_groups=[{"race":0}],
                                          privileged_groups=[{"race":1}]) 

metric_orig = BinaryLabelDatasetMetric(dataset_tf_train,
                                       unprivileged_groups=[{"race":0}],
                                       privileged_groups=[{"race":1}]) 
disparate_impact_orig = MetricTextExplainer(metric_orig).disparate_impact()
print(disparate_impact_orig)


# In[136]:


# APPLYING MODEL WITH REWEIGHNING 


# In[137]:


scale_tf = StandardScaler().fit(dataset_tf_train.features) 
X_train_tf = scale_tf.transform(dataset_tf_train.features)  
y_train = dataset_tf_train.labels.ravel()


# In[138]:


X_test_tf = scale_tf.transform(test_set_binary.features) 
y_test = test_set_binary.labels.ravel()


# In[139]:


#Running XGBOOST Model on the transformed data after implementing reweighing
#model_tf_rf = RandomForestClassifier(n_estimators=300,min_samples_split=10, max_depth=10,n_jobs=-1)
xgb_best.fit(X_train_tf, y_train,sample_weight=dataset_tf_train.instance_weights)


# In[140]:


#predicted score from RF model on the transformed (reweighed) dataset
y_pred_test_tf = xgb_best.predict(X_test_tf)
acc_test = accuracy_score(y_true=y_test, y_pred=y_pred_test_tf)
print("Test accuracy: ", acc_test)


# In[141]:


dataset_pred_tf = test_set.copy()
dataset_pred_tf['readmitted'] = y_pred_test_tf


# In[142]:


# Making our data into a Binary label dataset for prediction reweighed dataset
aif_fair_model_tf = BinaryLabelDataset(favorable_label = 1, unfavorable_label = 0, df=dataset_pred_tf,
                                                      label_names=["readmitted"],
                                                     protected_attribute_names=["race"],
                                              privileged_protected_attributes = [1,1])


# In[143]:


bin_metric_tf = BinaryLabelDatasetMetric(aif_fair_model_tf, 
                                     unprivileged_groups=[{"race":0}],
                                      privileged_groups=[{"race":1}]) 

disparate_impact_bin_tf = bin_metric_tf.disparate_impact()
print('Disparate impact after the reweighing method: ', disparate_impact_bin_tf)
mean_difference_tf = bin_metric_tf.mean_difference()
print('Mean difference after the reweighing method: ', mean_difference_tf)


# In[144]:


### After Reweighing, the disperate impact in the dataset is 0.97, so there is no bias in the model.
    ###But , if bias remains in the dataset, other in-processing mitigation algorithm should be applied.As mitigation should be performed early stages. I have checked with another algorithm disperate impact remover.


# In[145]:


#Applying  pre-process algorithm: 2 (Disperate impact remover)


# In[146]:


trainset_Binary = BinaryLabelDataset(favorable_label = 1, unfavorable_label = 0, df=train_set,
                                                      label_names=["readmitted"],
                                                     protected_attribute_names=["race"])


# In[147]:


from aif360.algorithms.preprocessing import DisparateImpactRemover
# Initialize the DisparateImpactRemover with the sensitive attribute
di_remover = DisparateImpactRemover(repair_level=1.0)

# Apply the transformation
dataset_transformed_train = di_remover.fit_transform(trainset_Binary)




# In[148]:


# Initialize the DisparateImpactRemover with the sensitive attribute
di_remover = DisparateImpactRemover(repair_level=1.0)

# Apply the transformation
dataset_transformed_train = di_remover.fit_transform(trainset_Binary)



# In[149]:


scale_DIRemoval = StandardScaler().fit(dataset_transformed_train.features) 
X_train_DIRemoval = scale_DIRemoval.transform(dataset_transformed_train.features)  
y_train_DIRemoval = dataset_transformed_train.labels.ravel()


# In[150]:


X_test_DIRemoval = scale_DIRemoval.transform(test_set_binary.features) 
y_test_DIRemoval = test_set_binary.labels.ravel()


# In[151]:


#Running XGB  Model on the transformed data after implementing reweighing
#model_DIRemoval_rf = RandomForestClassifier(n_estimators=300,min_samples_split=10, max_depth=10,n_jobs=-1)
model_DIRemoval_xgb=xgb_best.fit(X_train_DIRemoval, y_train_DIRemoval,sample_weight=dataset_transformed_train.instance_weights)


# In[152]:


#predicted score from RF model on the transformed (reweighed) dataset
y_pred_test_DIRemoval = model_DIRemoval_xgb.predict(X_test_DIRemoval)
acc_test = accuracy_score(y_true=y_test_DIRemoval, y_pred=y_pred_test_DIRemoval)
print("Test accuracy: ", acc_test)


# In[153]:


dataset_pred_DIRemoval = test_set.copy()
dataset_pred_DIRemoval['readmitted'] = y_pred_test_DIRemoval

dataset_pred_DIRemoval


# In[154]:


# Making our data into a Binary label dataset for prediction reweighed dataset
aif_fair_model_DIRemoval = BinaryLabelDataset(favorable_label = 1, unfavorable_label = 0, df=dataset_pred_DIRemoval,
                                                      label_names=["readmitted"],
                                                     protected_attribute_names=["race"],
                                              privileged_protected_attributes = [1,1])


# In[155]:


bin_metric_DIRemoval = BinaryLabelDatasetMetric(aif_fair_model_DIRemoval, 
                                     unprivileged_groups=[{"race":0}],
                                      privileged_groups=[{"race":1}]) 

disparate_impact_bin_DIremoval = bin_metric_DIRemoval.disparate_impact()
print('Disparate impact after the DI remover method: ', disparate_impact_bin_DIremoval)
mean_difference_DIremoval = bin_metric_DIRemoval.mean_difference()
print('Mean difference after the DI remover method: ', mean_difference_DIremoval)


# In[156]:


#As Disparity between privileged and unprivileged groups have increased, further pre-percessing or in-processing algorithms should be applied.


# ### Trying in process methods: 1.Adversial debiasing and prejudice remover (In-processing algorithm)

# In[157]:


trainset_Binary_adb = BinaryLabelDataset(favorable_label = 1, unfavorable_label = 0, df=train_set,
                                                      label_names=["readmitted"],
                                                     protected_attribute_names=["race"])
#trainset_Binary_final=trainset_Binary_adb.copy()
    


# In[158]:


import tensorflow as tf
tf.compat.v1.disable_eager_execution 


# In[159]:


from tensorflow.python.framework.ops import disable_eager_execution


# In[160]:


# Define the adversarial debiasing model
sess = tf.compat.v1.Session()

plain_model = AdversarialDebiasing(privileged_groups=[{'race': 1}],
                          unprivileged_groups=[{'race': 0}],
                          scope_name='plain_classifier',
                          debias=False,
                          sess=sess)


# In[161]:


disable_eager_execution()


# In[162]:


# Train the model
plain_model.fit(trainset_Binary_adb)


# In[163]:


#Apply the plain model to test data(to predict in testdata)
dataset_nodebiasing_train = plain_model.predict(trainset_Binary_adb)
dataset_nodebiasing_test = plain_model.predict(test_set_binary)


# In[164]:


# Metrics for the dataset from plain model (without debiasing)
#display(Markdown("#### Plain model - without debiasing - dataset metrics"))
metric_dataset_nodebiasing_train = BinaryLabelDatasetMetric(dataset_nodebiasing_train, 
                                            unprivileged_groups=[{'race': 0}],
                                            privileged_groups=[{'race': 1}])

print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_train.mean_difference())

metric_dataset_nodebiasing_test = BinaryLabelDatasetMetric(dataset_nodebiasing_test, 
                                             unprivileged_groups=[{'race':0}],
                                             privileged_groups=[{'race':1}])

print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_test.mean_difference())

#display(Markdown("#### Plain model - without debiasing - classification metrics"))


# In[165]:


classified_metric_nodebiasing_test = ClassificationMetric(test_set_binary, 
                                                 dataset_nodebiasing_test,
                                                 unprivileged_groups=[{'race':0}],
                                                 privileged_groups=[{'race':1}])
print("Test set: Classification accuracy = %f" % classified_metric_nodebiasing_test.accuracy())
TPR = classified_metric_nodebiasing_test.true_positive_rate()
TNR = classified_metric_nodebiasing_test.true_negative_rate()
bal_acc_nodebiasing_test = 0.5*(TPR+TNR)
print("Test set: Balanced classification accuracy = %f" % bal_acc_nodebiasing_test)
print("Test set: Disparate impact = %f" % classified_metric_nodebiasing_test.disparate_impact())
print("Test set: Equal opportunity difference = %f" % classified_metric_nodebiasing_test.equal_opportunity_difference())
print("Test set: Average odds difference = %f" % classified_metric_nodebiasing_test.average_odds_difference())
print("Test set: Theil_index = %f" % classified_metric_nodebiasing_test.theil_index())


# ### Now applying this method with debiasing parameter in trainset

# In[166]:


tf.compat.v1.disable_eager_execution
sess = tf.compat.v1.Session()


# In[167]:


trainset_Binary_withdb = BinaryLabelDataset(favorable_label = 1, unfavorable_label = 0, df=train_set,
                                                      label_names=["readmitted"],
                                                     protected_attribute_names=["race"])


# In[168]:


# Learn parameters with debias set to True
debiased_model = AdversarialDebiasing(privileged_groups = [{'race':1}],
                          unprivileged_groups = [{'race':0}],
                          scope_name='debiased_classifier',
                          debias=True,
                          sess=sess)


# In[169]:


#debiased_model.fit(trainset_Binary_withdb)


# In[170]:


debiased_model.fit(trainset_Binary_withdb)


# In[171]:


# Apply the plain model to test data
dataset_debiasing_train = debiased_model.predict(trainset_Binary_withdb)
dataset_debiasing_test = debiased_model.predict(test_set_binary)


# In[172]:


# Metrics for the dataset from plain model (without debiasing)
#display(Markdown("#### Plain model - without debiasing - dataset metrics"))
metric_dataset_withdebiasing_train = BinaryLabelDatasetMetric(dataset_debiasing_train, 
                                            unprivileged_groups=[{'race': 0}],
                                            privileged_groups=[{'race': 1}])

print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_train.mean_difference())

metric_dataset_withdebiasing_test = BinaryLabelDatasetMetric(dataset_debiasing_test, 
                                             unprivileged_groups=[{'race':0}],
                                             privileged_groups=[{'race':1}])

print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_test.mean_difference())

#display(Markdown("#### Plain model - without debiasing - classification metrics"))


# In[173]:


classified_metric_withdebiasing_test = ClassificationMetric(test_set_binary, 
                                                 dataset_debiasing_test,
                                                 unprivileged_groups=[{'race':0}],
                                                 privileged_groups=[{'race':1}])
print("Test set: Classification accuracy = %f" % classified_metric_withdebiasing_test.accuracy())
TPR = classified_metric_withdebiasing_test.true_positive_rate()
TNR = classified_metric_withdebiasing_test.true_negative_rate()
bal_acc_debiasing_test = 0.5*(TPR+TNR)
print("Test set: Balanced classification accuracy = %f" % bal_acc_debiasing_test)
print("Test set: Disparate impact = %f" % metric_dataset_withdebiasing_test.disparate_impact())
print("Test set: Equal opportunity difference = %f" % classified_metric_withdebiasing_test.equal_opportunity_difference())
print("Test set: Average odds difference = %f" % classified_metric_withdebiasing_test.average_odds_difference())
print("Test set: Theil_index = %f" % classified_metric_withdebiasing_test.theil_index())


# In[174]:


### PRejudice remover
from aif360.algorithms.inprocessing import PrejudiceRemover


# In[175]:


trainset_Binary_pr = BinaryLabelDataset(favorable_label = 1, unfavorable_label = 0, df=train_set,
                                                      label_names=["readmitted"],
                                                     protected_attribute_names=["race"])


# In[176]:


testset_Binary_pr = BinaryLabelDataset(favorable_label = 1, unfavorable_label = 0, df=test_set,
                                                      label_names=["readmitted"],
                                                     protected_attribute_names=["race"])


# In[177]:


# Initialize Prejudice Remover model
pr = PrejudiceRemover(sensitive_attr='race', eta=25.0)


# In[178]:


# Train the model
pr.fit(trainset_Binary_pr)


# In[179]:


# Make predictions
dataset_debiasing_train = pr.predict(trainset_Binary_pr)
dataset_debiasing_test = pr.predict(testset_Binary_pr)


# In[180]:


# Evaluate the model
classified_metric_train = ClassificationMetric(
    trainset_Binary_pr,
    dataset_debiasing_train,
    unprivileged_groups=[{'race': 0}],
    privileged_groups=[{'race': 1}]
)
classified_metric_test = ClassificationMetric(
    testset_Binary_pr,
    dataset_debiasing_test,
    unprivileged_groups=[{'race': 0}],
    privileged_groups=[{'race': 1}]
)


# In[181]:


# Calculate and print Disparate Impact and other fairness metrics
def print_fairness_metrics(metric, dataset_name):
    disparate_impact = metric.disparate_impact()
    statistical_parity_difference = metric.statistical_parity_difference()
    equal_opportunity_difference = metric.equal_opportunity_difference()
    average_odds_difference = metric.average_odds_difference()
    
    print(f"{dataset_name} Fairness Metrics:")
    print(f"  Disparate Impact: {disparate_impact:.4f}")
    print(f"  Statistical Parity Difference: {statistical_parity_difference:.4f}")
    print(f"  Equal Opportunity Difference: {equal_opportunity_difference:.4f}")
    print(f"  Average Odds Difference: {average_odds_difference:.4f}")
    print()


# In[182]:


# Metrics for original and debiased datasets
metric_orig_train = BinaryLabelDatasetMetric(
    trainset_Binary_pr,
    unprivileged_groups=[{'race': 0}],
    privileged_groups=[{'race': 1}]
)
metric_debias_train = ClassificationMetric(
    trainset_Binary_pr,
    dataset_debiasing_train,
    unprivileged_groups=[{'race': 0}],
    privileged_groups=[{'race': 1}]
)
metric_debias_test = ClassificationMetric(
    testset_Binary_pr,
    dataset_debiasing_test,
    unprivileged_groups=[{'race': 0}],
    privileged_groups=[{'race': 1}]
)

# Print fairness metrics
metric_orig_train.disparate_impact()
print("Test set: Disparate impact = %f" % metric_debias_test.disparate_impact())
# Print accuracy
print("Train set: Classification accuracy = {:.4f}".format(classified_metric_train.accuracy()))
print("Test set: Classification accuracy = {:.4f}".format(classified_metric_test.accuracy()))


# In[183]:


#applying SHAP: Explainable AI 


# In[184]:


import shap
shap.initjs()


# In[ ]:


# Initialize SHAP explainer
explainer = shap.TreeExplainer(xgb_best, X_train_scaled)

# Calculate SHAP values for the test set
shap_values = explainer(X_test_scaled)

# Plot the SHAP values
shap.summary_plot(shap_values, X_test_scaled)


# Feature value high= red, low=blue. The color of each feature indicates its value. This can be especially important in describing the models behavior and each feature values impact. Time in hospital is most important feature here. red colour indicates high no of time stayed in hospital.

# In[ ]:


# Load the California housing dataset
# Specify the name of the target column
target_column = 'readmitted'  # Replace 'target' with your actual target column name

# Split the dataset into features (X) and target (y)
X = raw_df.drop(columns=[target_column])
y = raw_df[target_column]


# In[ ]:


# Plot the SHAP values
shap.summary_plot(shap_values, plot_type="bar",feature_names=X.columns)


# In[ ]:


#Local Interpretability explainability of individual prediction that has been made by the model.


# In[ ]:


shap_df = pd.DataFrame(shap_values, columns=X, index=X_test_scaled.index)
idx = 123
shap.force_plot(explainer.expected_value, 
                shap_df.reset_index(drop=True).iloc[idx].values, 
                X_test_scaled.reset_index(drop=True).iloc[idx])


# In[ ]:


#Checking disperate impact on main dataset


# In[ ]:


train_set_latest=train_set.copy()
test_set_latest=test_set.copy()


# In[ ]:


test_set_latest


# In[ ]:


train_set_latest.to_excel("trainset_final.xlsx",index=False)
test_set_latest.to_excel("testset_final.xlsx",index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




