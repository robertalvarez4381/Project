# --- Step 1: imports ---

# Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Model Selection & Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Models
from sklearn.svm import SVC 
from sklearn.neural_network import MLPClassifier 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier 

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Configuration for cleaner charts
sns.set(style="whitegrid")
print("Libraries imported successfully.")

# --- Step 2: Data Loading and Cleaning ---
file_path = 'HR_comma_sep.csv'

data = pd.read_csv(file_path)
print(f"Successfully loaded '{file_path}'.")
    
# Rename columns for clarity
data = data.rename(columns={
    'satisfaction_level': 'SatisfactionLevel',
    'last_evaluation': 'LastEvaluation',
    'number_project': 'NumberofProjects',
    'average_montly_hours': 'AverageMonthlyHours',
    'time_spend_company': 'TimeSpentAtCompany',
    'Work_accident': 'WorkAccident',
    'left': 'Left', 
    'promotion_last_5years': 'PromotionsInLast5Years',
    'sales': 'Department',
    'salary': 'Salary'
})

data.head()

print("--- Data Structure ---")
data.info()

print("\n--- Missing Values ---")
print(data.isnull().sum())

# --- Step 3: EDA ---

# 3.1 Correlation Matrix
plt.figure(figsize=(8, 6))
correlation_matrix = data.select_dtypes(include=['float64', 'int64']).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='RdBu', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of HR Data')
plt.show()

# 3.2 Departmental Analysis: Sorted Volume vs. Rate
fig, axes = plt.subplots(1, 2, figsize=(20, 6))

# Chart 1: The Volume (Sorted by Size)
# 1. Calculate the order (Biggest departments first)
volume_order = data['Department'].value_counts().index

sns.countplot(
    y='Department', 
    hue='Left', 
    data=data, 
    palette='viridis',
    order=volume_order, 
    ax=axes[0]
)
axes[0].set_title('1. Turnover Volume')
axes[0].set_xlabel('Number of Employees')
axes[0].legend(title='Status', labels=['Stayed', 'Left'])


# Chart 2: The Rate (Sorted by Risk)
# Calculate rates
dept_counts = data.groupby(['Department', 'Left']).size().unstack()
dept_counts['Turnover_Rate'] = (dept_counts[1] / (dept_counts[0] + dept_counts[1])) * 100
dept_counts = dept_counts.sort_values(by='Turnover_Rate', ascending=False)

sns.barplot(
    x=dept_counts['Turnover_Rate'], 
    y=dept_counts.index,  
    ax=axes[1]
)
axes[1].set_title('2. Turnover Rate (Risk Percentage)')
axes[1].set_xlabel('Turnover Rate (%)')
axes[1].set_ylabel('') 

# Add average line
avg_turnover = (data['Left'].mean() * 100)
axes[1].axvline(avg_turnover, color='red', linestyle='--', label=f'Avg: {avg_turnover:.1f}%')
axes[1].legend()

plt.tight_layout()
plt.show()

# ### Salary Level Distribution and Turnover Risk

# %%
# 3.3 Salary Analysis: Volume vs. Rate
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Define the logical order for salary (Low -> High)
salary_order = ['low', 'medium', 'high']

# Chart 1: The Volume (Count)
sns.countplot(
    x='Salary', 
    hue='Left', 
    data=data, 
    order=salary_order, 
    palette='viridis',
    ax=axes[0]
)
axes[0].set_title('1. Salary Turnover Volume (Headcount)')
axes[0].set_xlabel('Salary Level')
axes[0].set_ylabel('Number of Employees')
axes[0].legend(title='Status', labels=['Stayed', 'Left'])


# Chart 2: The Rate (Percentage)
# 1. Calculate rates
salary_counts = data.groupby(['Salary', 'Left']).size().unstack()
salary_counts['Turnover_Rate'] = (salary_counts[1] / (salary_counts[0] + salary_counts[1])) * 100

# 2. Reindex to ensure the bars are in the correct Low -> High order
salary_counts = salary_counts.reindex(salary_order)

sns.barplot(
    x=salary_counts.index, 
    y=salary_counts['Turnover_Rate'], 
    order=salary_order,
    ax=axes[1]
)
axes[1].set_title('2. Salary Turnover Rate (Risk Percentage)')
axes[1].set_xlabel('Salary Level')
axes[1].set_ylabel('Turnover Rate (%)')

# Add average line for context
avg_turnover = (data['Left'].mean() * 100)
axes[1].axhline(avg_turnover, color='red', linestyle='--', label=f'Avg: {avg_turnover:.1f}%')
axes[1].legend()

plt.tight_layout()
plt.show()

# Print the exact numbers to confirm
print("Turnover Rate by Salary:")
print(salary_counts['Turnover_Rate'].to_markdown(floatfmt=".1f"))

# 3.4 Turnover Rate by Tenure
plt.figure(figsize=(10, 6))

# Calculate rates
tenure_metrics = data.groupby('TimeSpentAtCompany')['Left'].mean().reset_index()
tenure_metrics['Left'] = tenure_metrics['Left'] * 100 # Convert to percentage

sns.barplot(x='TimeSpentAtCompany', y='Left', data=tenure_metrics)

plt.title('Turnover Rate by Years')
plt.xlabel('Years at Company')
plt.ylabel('Turnover Rate (%)')
plt.axhline(data['Left'].mean()*100, color='red', linestyle='--', label='Company Avg')
plt.legend()
plt.show()

# 3.5 Graph: Turnover vs. Number of Projects
plt.figure(figsize=(10, 6))

# We use a pointplot here to show the "U-Shaped" curve clearly
sns.pointplot(x='NumberofProjects', y='Left', data=data, color='purple', errorbar=None)

plt.title('Turnover Rate by Number of Projects')
plt.xlabel('Number of Projects')
plt.ylabel('Probability of Leaving')
plt.grid(True, alpha=0.3)
plt.show()

# 3.6 Graph: Projects vs. Monthly Hours (Aggregated)
plt.figure(figsize=(10, 6))
# Removing the 'hue' argument aggregates the data for each project level
sns.boxplot(
    x='NumberofProjects', 
    y='AverageMonthlyHours', 
    data=data, 
    color='tab:blue'
)
plt.title('Aggregated Monthly Hours Distribution by Number of Projects (Company Benchmark)')
plt.xlabel('Number of Projects')
plt.ylabel('Average Monthly Hours')
plt.show()

# 3.7 Satisfaction Level Distribution (KDE Plot)
plt.figure(figsize=(10, 6))

sns.kdeplot(data=data[data['Left'] == 1]['SatisfactionLevel'], label='Left', fill=True, color='red', alpha=0.3)
sns.kdeplot(data=data[data['Left'] == 0]['SatisfactionLevel'], label='Stayed', fill=True, color='blue', alpha=0.3)

plt.title('Distribution of Satisfaction Level: Stayed vs. Left')
plt.xlabel('Satisfaction Level')
plt.ylabel('Density')
plt.legend()
plt.show()

# 3.8 Side-by-Side EDA Comparison: Satisfaction vs Evaluation/Hours

# 1. Create a temporary column with readable labels
data['Status'] = data['Left'].map({0: 'Stayed', 1: 'Left'})

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Define colors: Stayed = Blue, Left = Red
custom_palette = {'Stayed': 'blue', 'Left': 'red'}

# Chart 1: Satisfaction vs. Evaluation (Left Side)
sns.scatterplot(
    x='SatisfactionLevel', 
    y='LastEvaluation', 
    hue='Status', 
    data=data, 
    palette=custom_palette, 
    alpha=0.4, 
    s=40,
    ax=axes[0]
)
axes[0].set_title('Satisfaction vs. Last Evaluation')
axes[0].set_xlabel('Satisfaction Level')
axes[0].set_ylabel('Last Evaluation Score')
# Force legend to bottom left
axes[0].legend(loc='lower left', title='Status', frameon=True)


# Chart 2: Satisfaction vs. Monthly Hours (Right Side)
sns.scatterplot(
    x='SatisfactionLevel', 
    y='AverageMonthlyHours', 
    hue='Status', 
    data=data, 
    palette=custom_palette, 
    alpha=0.4, 
    s=40,
    ax=axes[1]
)
axes[1].set_title('Satisfaction vs. Average Monthly Hours')
axes[1].set_xlabel('Satisfaction Level') 
axes[1].set_ylabel('Average Monthly Hours')
axes[1].legend(loc='lower left', title='Status', frameon=True)


plt.tight_layout()
plt.show()

# Clean up: Drop the helper column so it doesn't duplicate data later
data.drop('Status', axis=1, inplace=True)

# Step 4: Data Preprocessing 
print("--- Step 4: Data Preprocessing ---")

# Define our target variable (y) and features (X)
target = 'Left'

# We only drop the target itself ('Left')
features = data.drop([target], axis=1) 
y = data[target]

# Identify numerical and categorical features
numeric_features = [
    'SatisfactionLevel', 
    'LastEvaluation', 
    'NumberofProjects', 
    'AverageMonthlyHours', 
    'TimeSpentAtCompany', 
    'PromotionsInLast5Years',
    'WorkAccident'
]

categorical_features = ['Department', 'Salary']

print(f"Target variable: {target}")
print(f"Numeric features: {numeric_features}")
print(f"Categorical features: {categorical_features}\n")

# Create a preprocessing pipeline
# 1. For numeric features: Scale them
numeric_transformer = StandardScaler()

# 2. For categorical features: One-Hot Encode them
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Use ColumnTransformer to apply transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

print("Preprocessing pipeline created.")
print("-" * 40 + "\n")

# Model Training and Evaluation

# 1. Train-Test Split
# Step 5: Split Data 
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set:  {X_test.shape[0]} samples")

#  Step 5.5: Model Implementation

models = {
    "SVM": SVC(kernel='linear', random_state=42),
    "MLP (Neural Net)": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

results = {}

for name, model in models.items():
    print(f"Training {name}...")
    
    # Pipeline: Preprocess -> Train
    clf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    clf_pipeline.fit(X_train, y_train)
    y_pred = clf_pipeline.predict(X_test)
    
    # Metrics
    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred)
    }

print("\nTraining Complete.")

# Step 6: Model Comparison
results_df = pd.DataFrame(results).T

# Visualization
plt.figure(figsize=(10, 6))
sns.barplot(x=results_df['F1-Score'], y=results_df.index)
plt.title('Model Precision Comparison')
plt.xlim(0, 1.0)
plt.show()

# Display Table
print("Detailed Performance Metrics:")
display(results_df) #prints table

# Visualization of Prediction Results

# --- Visualization: Confusion Matrix (Random Forest) ---
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

print("--- Visualizing Confusion Matrix for Random Forest ---")

# build and train the Random Forest pipeline 
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

rf_pipeline.fit(X_train, y_train)

# 1. Generate predictions using the fitted Random Forest pipeline from Step 6
y_pred_rf = rf_pipeline.predict(X_test)

# 2. Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred_rf)

# 3. Plot the matrix
plt.figure(figsize=(8, 6))
# display_labels maps 0 to 'Stayed' and 1 to 'Left' for clarity
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Stayed', 'Left'])
disp.plot(cmap='Blues', values_format='d') # 'd' formats numbers as integers (no scientific notation)

plt.title('Confusion Matrix: Random Forest')
plt.grid(False) # Turn off grid lines for cleaner look
plt.show()

# Step 7: Feature Importance Analytics
print("--- Step 7: Feature Importance Analysis (from Random Forest) ---")

# Extract Feature Names 
# 1. Get the categorical names after OneHotEncoding
cat_feature_names = rf_pipeline.named_steps['preprocessor'] \
                               .named_transformers_['cat'] \
                               .get_feature_names_out(categorical_features)

# 2. Combine with the numeric features
all_feature_names = numeric_features + list(cat_feature_names)

# Extract Importances
importances = rf_pipeline.named_steps['classifier'].feature_importances_

# Create a DataFrame for easy viewing
feature_importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Display the Top 10
print("Top 10 factors influencing employee retention:")
print(feature_importance_df.head(10).to_markdown(index=False, floatfmt=".4f"))

# Visualization: Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
plt.title('Top 10 Features Influencing Employee Retention')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# ## Step 8: Prediction Probabilities
print("--- Step 8: Employee Risk Scoring ---")

# 1. Get probability scores
risk_scores = rf_pipeline.predict_proba(X_test)[:, 1]

# 2. Create a DataFrame to analyze these scores
risk_df = X_test.copy()
risk_df['Risk_Score'] = risk_scores
risk_df['Actual_Status'] = y_test 

# Filter for ONLY Current Employees
# We only want to look at people who are currently at the company (Actual_Status == 0)
current_employee_risk = risk_df[risk_df['Actual_Status'] == 0].copy()

# 3. Segment employees into groups (Low, Medium, High Risk)
def categorize_risk(score):
    if score < 0.3:
        return 'Low Risk'
    elif score < 0.7:
        return 'Medium Risk'
    else:
        return 'High Risk'

current_employee_risk['Risk_Group'] = current_employee_risk['Risk_Score'].apply(categorize_risk)

# 4. Display ALL "High Risk" Current Employees
print("High Risk Employees:")
print(current_employee_risk[current_employee_risk['Risk_Group'] == 'High Risk'].sort_values(by='Risk_Score', ascending=False).to_markdown())

# 5. Analyze the High Risk Group
high_risk_employees = current_employee_risk[current_employee_risk['Risk_Group'] == 'High Risk']
print(f"\nNumber of CURRENT employees identified as High Risk: {len(high_risk_employees)}")

# Step 8: Global Cluster Analysis
from sklearn.cluster import KMeans

# 1. Select Features for Clustering
# We use these three because they visually define the "Leaver" profiles best
cluster_features = ['SatisfactionLevel', 'LastEvaluation', 'AverageMonthlyHours']
X_clustering = data[cluster_features].copy()

# 2. Scale the data (Crucial for K-Means)
cluster_scaler = StandardScaler()
X_scaled = cluster_scaler.fit_transform(X_clustering)

# 3. Fit K-Means & Create the Column
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
data['Archetype_Cluster'] = kmeans.fit_predict(X_scaled)

# Step 8: Risk Scoring & Action Plan Chart
print("--- Step 9: Risk Scoring Current Employees ---")

# 1. Isolate Current Employees
current_employee_mask = data['Left'] == 0
current_employees = data[current_employee_mask].copy()

# 2. Prepare Data for Prediction (Safety Step)
# We filter to only the columns the model was trained on to prevent errors
training_features = list(rf_pipeline.named_steps['preprocessor'].feature_names_in_)
X_current_clean = current_employees[training_features]

# 3. Generate Probabilities
risk_probs = rf_pipeline.predict_proba(X_current_clean)[:, 1]

# 4. Assign Scores back to the Dataframe
data.loc[current_employee_mask, 'Risk_Score'] = risk_probs
data['Risk_Score'] = data['Risk_Score'].fillna(0)

# 5. Identify "High Risk" Current Employees (> 70%)
high_risk_mask = (data['Left'] == 0) & (data['Risk_Score'] > 0.7)
print(f"High Risk Current Employees identified: {high_risk_mask.sum()}")

# 6. Chart 2: The Action Plan (Red Dots)
plt.figure(figsize=(12, 8))

# Layer 1: Background Context (Faint)
sns.scatterplot(
    x='SatisfactionLevel', 
    y='LastEvaluation', 
    hue='Archetype_Cluster', 
    data=data, 
    palette='viridis', 
    s=50, 
    alpha=0.2,            # Faded background
    legend=False
)

# Layer 2: High Risk Current Employees (Bright Red)
sns.scatterplot(
    x='SatisfactionLevel',
    y='LastEvaluation',
    data=data[high_risk_mask],
    color='red',
    marker='o',
    s=150,                # Big dots
    edgecolor='black',
    label='High Risk Current Employee (>70%)'
)

plt.title('Action Plan: High-Risk Employees Mapped to Turnover Archetypes')
plt.xlabel('Satisfaction Level')
plt.ylabel('Last Evaluation')
plt.legend(loc='upper left')
plt.show()

# Display the High Risk list
print("Actionable List (High Risk):")
cols_to_show = ['Department', 'SatisfactionLevel', 'LastEvaluation', 'Risk_Score']
print(data[high_risk_mask].sort_values('Risk_Score', ascending=False)[cols_to_show].head(5).to_markdown())
