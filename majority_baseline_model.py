import json
from sklearn.metrics import accuracy_score
import numpy as np

def load_dataset(file_path):
    # Load the JSON dataset
    with open(file_path, 'r') as file:
        dataset = json.load(file)
    return dataset

def majority_baseline_predict(labels):
    # Convert string labels to numerical values
    label_mapping = {label: index for index, label in enumerate(np.unique(labels))}
    print("labels", label_mapping)

    numerical_labels = np.array([label_mapping[label] for label in labels])
    
    # Find the majority class
    majority_class = np.argmax(np.bincount(numerical_labels))
    print("majority class is:", majority_class)
    
    # Predict the majority class for all instances
    predictions = np.full_like(numerical_labels, fill_value=majority_class)
    
    return predictions


# Replace 'your_dataset.json' with the actual file path of your JSON dataset
dataset_path = 'your_dataset.json'

# Load the dataset
dataset = load_dataset(dataset_path)

# Extract labels from the dataset and convert to numerical values
true_labels = np.array([entry['g'] for entry in dataset])

label_mapping = {label: index for index, label in enumerate(np.unique(true_labels))}

numerical_labels = np.array([label_mapping[label] for label in true_labels])

# Generate predictions using the majority baseline model
predicted_labels = majority_baseline_predict(true_labels)

# Calculate accuracy
accuracy = accuracy_score(numerical_labels, predicted_labels)

print(f"Accuracy of Majority Baseline Model on {dataset_path} is: {accuracy:.2%}")


