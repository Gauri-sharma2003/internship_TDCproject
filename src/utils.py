import pandas as pd
import json
from datetime import datetime

from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


import matplotlib.pyplot as plt
import seaborn as sns

import csv

import pickle



def clean_data(data, label_encode_cols=[], onehot_encode_cols=[]):
    # Handle missing values
    data.fillna(0, inplace=True)  # Example: fill missing values with 0
    
    # Encode categorical variables using LabelEncoder
    for col in label_encode_cols:
        data[col] = LabelEncoder().fit_transform(data[col])
    
    # Encode categorical variables using OneHotEncoder
    for col in onehot_encode_cols:
        encoded_cols = pd.get_dummies(data[col], prefix=col)
        data = pd.concat([data, encoded_cols], axis=1)
        data.drop(col, axis=1, inplace=True)
    
    # Scale numerical features
    numerical_cols = data.select_dtypes(include=['int', 'float']).columns
    data[numerical_cols] = StandardScaler().fit_transform(data[numerical_cols])
    
    return data

def prediction_data_clean_pipeline(input_features):
    scaler = MinMaxScaler()
    scaled_features = scaler.transform(input_features)
    return scaled_features


def load_model_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model


def remove_outliers_and_plot(df, columns):
    print("_"*30)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    print("Shape Before Removing Outlier : ",df.shape)
    # Draw box plot before removing outliers
    df.boxplot(column=columns, ax=axes[0])
    axes[0].set_title('Before Removing Outliers')
    for column in columns:
        
        
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    # Draw box plot after removing outliers
    df.boxplot(column=columns, ax=axes[1])
    axes[1].set_title('After Removing Outliers')

    plt.show()
    print("Shape After Removing Outlier : ",df.shape)
    print("_"*30)
    return df


def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Confusion Matrix': cm
    }
    
def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'Neural Network': MLPClassifier()
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
#         accuracy = accuracy_score(y_test, y_pred)
#         precision = precision_score(y_test, y_pred)
#         recall = recall_score(y_test, y_pred)
#         f1 = f1_score(y_test, y_pred)
        
        cr = classification_report(y_test, y_pred)
        
        print("_"*30)
        print("Model Name: ",name)
        print("CLassification Report",cr)
        
        results[name]=cr
    
#     results_df = pd.DataFrame(results)
#     best_model = results_df.loc[results_df['F1 Score'].idxmax()]
    
    return results

def draw_box_plot(df, columns,show_outliers=True):
    data = [df[col] for col in columns]
    title=f"Box Plot"
    xlabel=''
    ylabel=''
    plt.figure(figsize=(8, 6))
    plt.boxplot(data, showfliers=show_outliers)
    plt.xticks(range(1, len(columns) + 1), columns)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()
    

# Log visits to CSV file
def log_visit(page_name):
    with open('visit_log.csv', 'a', newline='') as csvfile:
        fieldnames = ['timestamp','page']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write headers if file is empty
        if csvfile.tell() == 0:
            writer.writeheader()

        # Write current visit information
        writer.writerow({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'page':page_name
        })