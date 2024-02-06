import os
import json
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

# Assuming the datasets directory is a sibling to the script location
# and plots will be saved to a 'plots' directory also at the same level as the script
datasets_dir = 'datasets'
plots_dir = 'plots'

# Use the MacOSX backend or adjust for a more generic backend compatible with all OS like 'Agg'
matplotlib.use('MacOSX')  

# Ensure plots directory exists 
os.makedirs(plots_dir, exist_ok=True)


# Load and process all the different datasets
def load_and_process_dataset(file_path):
    if file_path is None:
        return None
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Train and evaluate the models on decision tree classifier
def train_and_evaluate_model(train_data, test_data, dataset_name):
    # Calculate features for the datasets
    train_features = calculate_features(train_data)
    test_features = calculate_features(test_data)

    # Extract labels
    train_labels = [item['g'] for item in train_data]
    test_labels = [item['g'] for item in test_data]

    # Convert to DataFrame
    train_features_df = pd.DataFrame(train_features)
    test_features_df = pd.DataFrame(test_features)

    # Train-test split for validation
    X_train, X_test, y_train, y_test = train_test_split(train_features_df, train_labels, test_size=0.2, random_state=42)

    # Train classifier
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    
    # Validation evaluation
    y_pred = clf.predict(X_test)
    validate_model_performance(y_test, y_pred, f"{dataset_name} - Validation Set")
    cm_validation = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm_validation, dataset_name, 'Validation Set')
    
    # Test set evaluation
    y_test_pred = clf.predict(test_features_df)
    validate_model_performance(test_labels, y_test_pred, f"{dataset_name} - Test Set")
    cm_test = confusion_matrix(test_labels, y_test_pred)
    plot_confusion_matrix(cm_test, dataset_name, 'Test Set')

# Function to create a classification report per data set
def validate_model_performance(y_true, y_pred, dataset_name):
    unique_labels = np.unique(np.concatenate((y_true, y_pred)))
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=unique_labels, target_names=unique_labels, zero_division=0)
    
    print(f"{dataset_name} Accuracy:")
    print(f"Accuracy: {accuracy}")
    print(f"{dataset_name} Performance:")
    print(report)

# Function to print confusion matrices with unique labels per data set
def print_confusion_matrices(y_true, y_pred, dataset_name):
    unique_labels = np.unique(np.concatenate((y_true, y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    cm_df = pd.DataFrame(cm, index=unique_labels, columns=unique_labels)
    
    print(f"{dataset_name} Confusion Matrix:")
    print(cm_df)
    print("\n")

# Function to create plot from confusion matrix
def plot_confusion_matrix(cm, dataset_name, dataset_type):
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.title(f'Confusion Matrix for {dataset_name} - {dataset_type}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    # Save the plot with dynamic filename
    # and save plot to the 'plots' directory
    plt.savefig(os.path.join(plots_dir, f'{dataset_name}-{dataset_type}.png')) 
    # Display plot
    plt.show(block=True)
    

# Function to calculate word overlap features
def calculate_features(data):
    vectorizer = CountVectorizer(binary=True)

    # Calculate features for each data point
    features = []
    for item in data:
        premise = item['p']
        hypothesis = item['h']

        # Remove determiners and tokenize 
        premise_words = remove_determiners(premise.split())
        hypothesis_words = remove_determiners(hypothesis.split())

        # Transform without determiners
        X = vectorizer.fit_transform([''.join(premise_words), ''.join(hypothesis_words)]).toarray()

        # Word Overlap Count
        overlap_count = np.sum(np.min(X, axis=0))

        # Jaccard Similarity
        jaccard_sim = jaccard_similarity(premise_words, hypothesis_words)

        # Premise Word Overlap
        premise_word_count = len(premise_words)
        premise_overlap = overlap_count / premise_word_count if premise_word_count > 0 else 0

        # List of common negation words
        neg_words = {"n't", 'no', 'not', 'nobody', 'not', 'nor', 'never', 'nothing', 'none'}

        # Count negation words in premise and hypothesis    
        neg_count_premise = sum(word in neg_words for word in premise_words)
        neg_count_hypothesis = sum(word in neg_words for word in hypothesis_words)

        # Calculate negation_xor feature
        if abs(neg_count_premise - neg_count_hypothesis) % 2 == 0:
            negation_xor = 0
        else:
            negation_xor = 1

        features.append({
            'overlap_count': overlap_count, 
            'jaccard_similarity': jaccard_sim,
            'premise_overlap': premise_overlap,
            'negation_xor' : negation_xor
        })
    return features

# List of common determiners
determiners = set(['the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her', 'its', 'our', 'their'])

# Function to remove determiners from a list of words
def remove_determiners(word_list):
    return [word for word in word_list if word not in determiners]

# Function to calculate Jaccard Similarity
def jaccard_similarity(list1, list2):
    list1 = remove_determiners(list1)
    list2 = remove_determiners(list2)
    intersection = len(set(list1).intersection(set(list2)))
    union = len(set(list1)) + len(set(list2)) - intersection
    return float(intersection) / union

def main():
    # List of all the directories to the different data sets to load 
    datasets = [
        # Replace 'your_train_dataset.json' and 'your_test_dataset.json' with the actual file path of your JSON dataset
        ('your_train_dataset.json', 'your_test_dataset.json', 'Negation SNLI Data'),
        ('your_train_dataset.json', 'your_test_dataset.json', 'Negation MNLI Data'),
        ('your_train_dataset.json', 'your_test_dataset.json', 'Negation SICK Data'),
        ('your_train_dataset.json', 'your_test_dataset.json', 'Negation HANS Data'),
        ('your_train_dataset.json', 'your_test_dataset.json', 'Negation ANLI Data'),
        ('your_train_dataset.json', 'your_test_dataset.json', 'Negation WANLI Data')
        ('your_train_dataset.json', 'your_test_dataset.json', 'Negation FNLI Data')
    ]

    for train_path, test_path, dataset_name in datasets:
        print(f"Processing {dataset_name}")
        train_data = load_and_process_dataset(train_path)
        test_data = load_and_process_dataset(test_path)
        train_and_evaluate_model(train_data, test_data, dataset_name)

if __name__ == "__main__":
    main()
