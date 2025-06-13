# 🎓 Student Lifestyle Clustering

This project applies unsupervised machine learning (clustering) techniques to segment university students based on lifestyle-related features. The goal is to help universities understand student behavior better and develop tailored wellness, academic, and engagement strategies.

## 🧠 Project Overview

Universities often use one-size-fits-all approaches for student services. This project uses clustering to identify distinct student lifestyle patterns—such as study habits, sleep, exercise, and social activity levels—to help institutions better serve their diverse student populations.

### 📊 Dataset Description

The dataset was synthetically generated to simulate realistic student behaviors. It contains the following features:

- `Age`
- `Gender`
- `Study Hours per Week`
- `Exercise Frequency (per week)`
- `Sleep Hours per Night`
- `Cafeteria Spend ($/week)`
- `Social Activity Score (1–100)`

> 💡 Gender was encoded numerically during preprocessing for modeling purposes.

### 🎯 Objectives

- Perform data cleaning and encoding.
- Apply Agglomerative Clustering to group students.
- Visualize and interpret the clusters.
- Provide insights for actionable outcomes.

### 🛠️ Tools & Libraries

- Python
- Pandas
- NumPy
- scikit-learn (Agglomerative Clustering)
- Matplotlib (for visualization)

### 📌 Key Steps

1. Data Preprocessing (e.g., encoding gender)
2. Applying Agglomerative Clustering
3. Visualizing Clusters (Annual Income vs Spending Score as a similar analogy)
4. Interpreting Cluster Behavior

### 📈 Sample Insight

Clusters may reveal segments like:

- 🧘 Health-focused students (high exercise, moderate study, balanced sleep)
- 🎓 Academic achievers (high study hours, low social score)
- 🕺 Socially active students (high social activity, low study)
- 😴 Possibly at-risk students (low in all metrics)

### 📁 File Structure

