# 📊 HR Retention and Attrition Analysis

## Project Overview

This project focuses on identifying the key drivers of employee turnover (attrition) and building a machine learning model to predict which current employees are at the highest risk of leaving.

The analysis moves beyond simple correlation to define distinct employee "archetypes" and provides a targeted, actionable plan for HR to retain top talent.

## 🚀 Key Insights and Executive Summary

The analysis revealed that employee turnover is not a single issue but is driven by **three distinct structural failures**, visible as clusters in the data:

1.  **🔥 The Burned-Out:** High-performing employees who are severely overworked ($AverageMonthlyHours > 250$, $Satisfaction < 0.2$).
2.  **😴 The Underperformers:** Employees with low engagement, low evaluation scores, and low hours ($LastEvaluation < 0.55$, $AverageMonthlyHours < 160$).
3.  **🌟 The Superstars (Happy Leavers):** Highly productive and satisfied employees who leave, likely due to external factors like better compensation from competitors ($Satisfaction > 0.7$, $LastEvaluation > 0.8$).

---

### 🎯 Actionable Recommendation: Targeted Intervention Plan

Using the **Random Forest Model (Accuracy: 99.03%)**, we applied the prediction to the current workforce and identified **3 specific employees** who hold a **>70% probability of leaving**.

| Segment | Employees Identified | Action Plan |
| :--- | :--- | :--- |
| **Superstars** | 2 | **Immediate Retention:** Review compensation and career paths. Conduct a "Stay Interview" this week. |
| **Mismatched (Underperformer)** | 1 | **Performance Management:** Place on a Performance Improvement Plan (PIP). Avoid spending retention budget here. |

## 📊 Exploratory Data Analysis (EDA) Highlights

The following factors were found to be the most critical in predicting employee attrition:

| Feature | Insight | Recommendation |
| :--- | :--- | :--- |
| **Satisfaction Level** | **Highest Correlation** with turnover. The tri-modal distribution is the foundation for the "archetype" segmentation. | Address the root cause for each distinct cluster. |
| **Salary Level** | Employees in the **Low salary** bracket have the highest turnover rate (**29.7%**), significantly higher than Medium (**20.4%**). | Prioritize salary review for low-paid employees. |
| **Time Spent at Company** | Turnover spikes sharply between **Years 4 and 6**, suggesting a critical retention intervention window around **Year 3**. | Implement proactive retention programs (promotions, raises, rotation) for employees entering their 3rd year. |
| **Number of Projects** | The **"Goldilocks Zone"** for retention is **3 to 5 projects**. Too few leads to boredom; too many leads to burnout. | Managers should aim to keep employee workload between 3 and 5 projects to reduce burnout risk. |

## 🤖 Modeling and Prediction

### 1. Model Performance Comparison

The Random Forest Classifier provided the superior balance of metrics, making it the chosen model for final predictions.

| Model | Accuracy | Precision | Recall | **F1-Score** |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | **0.9903** | **0.9957** | **0.9636** | **0.9794** |
| Decision Tree | 0.9733 | 0.9261 | 0.9650 | 0.9451 |

### 2. Feature Importance

The Random Forest model identified the most influential factors in predicting whether an employee will leave:

1.  **`SatisfactionLevel`** ($0.3023$ importance)
2.  **`NumberofProjects`** ($0.1803$ importance)
3.  **`TimeSpentAtCompany`** ($0.1795$ importance)
4.  `AverageMonthlyHours` ($0.1631$ importance)
5.  `LastEvaluation` ($0.1270$ importance)

## 💻 Repository Contents

| File | Description |
| :--- | :--- |
| `HR_Retention.ipynb` | The primary Jupyter Notebook containing all data cleaning, EDA, visualization, modeling, and final risk scoring. |
| `HR_Retention.py` | A clean Python script version of the notebook for deployment or review. |
| `HR_comma_sep.csv` | The raw dataset used for the analysis. |
