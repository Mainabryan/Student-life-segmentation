import streamlit as st

st.title("Student Life Segmentation")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering

data = pd.read_csv('/content/student_lifestyle_data.csv')

data.head()

data.tail()

data.isnull().sum()

data.columns

import seaborn as sns

sns.boxplot(data['Study Hours per Week'])

import seaborn as sns
import matplotlib.pyplot as plt

numeric_features = ['Age', 'Gender', 'Study Hours per Week',
       'Exercise Frequency (per week)', 'Sleep Hours per Night',
       'Cafeteria Spend ($/week)', 'Social Activity Score (1–100)']

plt.figure(figsize=(14, 12))

for i, feature in enumerate(numeric_features):
    plt.subplot(4, 2, i+1)
    sns.boxplot(x=data[feature], color="skyblue")
    plt.title(f"Box Plot of {feature}", fontsize=11)
    plt.xlabel("")

plt.tight_layout()
plt.show()


# Convert gender into numercal figures
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})


data.corr()

# Compute the correlation matrix
corr_matrix = data.corr()

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Draw the heatmap
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

plt.title("Correlation Heatmap")
plt.show()

sns.histplot(data["Sleep Hours per Night"])
sns.boxplot(x=data["Cafeteria Spend ($/week)"])


data.info()

data.describe()

sns.histplot(data["Sleep Hours per Night"], kde=True, color="skyblue")
plt.title("Distribution of Sleep Hours")
plt.show()

sns.boxplot(x=data["Cafeteria Spend ($/week)"], color="salmon")
plt.title("Boxplot of Cafeteria Spend")
plt.show()

sns.histplot(data["Sleep Hours per Night"], bins=10, kde=True, color="skyblue")
plt.title("Distribution of Sleep Hours")
plt.xlabel("Sleep Hours per Night")
plt.ylabel("Number of Students")
plt.show()

plt.figure(figsize=(16, 12))

for i, feature in enumerate(numeric_features):
    plt.subplot(4, 2, i+1)
    sns.histplot(data[feature], kde=True, bins=20, color="skyblue")
    plt.title(f"Histogram + KDE of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

from sklearn.preprocessing import StandardScaler
features = ['Age', 'Gender', 'Study Hours per Week',
       'Exercise Frequency (per week)', 'Sleep Hours per Night',
       'Cafeteria Spend ($/week)', 'Social Activity Score (1–100)']


# initialize the scaler
scaler = StandardScaler()

# fit and transform the data
data[features] = scaler.fit_transform(data[features])

from sklearn.cluster import KMeans

# Initialize the KMeans model
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit the model to the data
kmeans.fit(data[features])

import matplotlib.pyplot as plt

inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data[features])
    inertia.append(kmeans.inertia_)

# Plotting the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o', linestyle='--', color='teal')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
plt.title('Elbow Method For Optimal k')
plt.grid(True)
plt.show()

from sklearn.cluster import KMeans

# Final KMeans with chosen k (e.g., 4)
kmeans_final = KMeans(n_clusters=6, random_state=42)
clusters = kmeans_final.fit_predict(data[features])

# Add cluster labels to the original data
data['Cluster'] = clusters

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(data['Study Hours per Week'],
            data['Social Activity Score (1–100)'],
            c=data['Cluster'], cmap='Set1', s=60)

plt.xlabel('Study Hours per Week')
plt.ylabel('Social Activity Score')
plt.title('Student Lifestyle Clusters')
plt.colorbar(label='Cluster Label')
plt.grid(True)
plt.show()


data.groupby('Cluster').mean(numeric_only=True)
