# import libraries
import numpy as np
import json

# Replace train and test datasets with the actual file paths of your JSON datasets
input_file_path_1 = 'your_train_data.json'
input_file_path_2 = 'your_test_data.json'

with open(input_file_path_1, 'r') as json_file:
    train_data = json.load(json_file)

with open(input_file_path_2, 'r') as json_file:
    test_data = json.load(json_file)

# Preprocess data for hypothesis-only bias
X_train_hypothesis = [my_dict['h'] for my_dict in train_data]
X_test_hypothesis = [my_dict['h'] for my_dict in test_data]
y_train = [my_dict['g'] for my_dict in train_data]
y_test = [my_dict['g'] for my_dict in test_data]


X_train_hypothesis = X_train_hypothesis[:20000]
X_test_hypothesis = X_test_hypothesis[:5000]
y_train = y_train[:20000]
y_test = y_test[:5000]

# function to calculate HO-features
def features(X_train_hypothesis, y_train):
    # count probabilities of label occuring in the dataset and make dictionary to save word occurences per label
    class_probs = {label: y_train.count(label) / len(y_train) for label in set(y_train)}
    word_class_counts = {label: {} for label in set(y_train)}
    total_words = {}

    # count occurence of all words and save to dictionary
    for hypothesis, label in zip(X_train_hypothesis, y_train):
        for word in hypothesis.split():
            if word not in total_words:
                total_words[word] = 0
            total_words[word] += 1
            
            # same for word occurence per label
            if word not in word_class_counts[label]:
                word_class_counts[label][word] = 0
            word_class_counts[label][word] += 1

    # calculate PMI score for each word per label
    pmi = {}
    for label in y_train:
        pmi[label] = {}
        for word in word_class_counts[label]:
            # only calculate pmi scores for words that frequently occur
            if total_words.get(word) >= 100:
                # Calculate probabilities for class, word, and word per class
                prob_class = class_probs[label]
                prob_word = total_words.get(word, 0.0) / sum(total_words.values())
                prob_wc = (word_class_counts[label].get(word, 0.0)) / (sum(word_class_counts[label].values()))
                pmi[label][word] = max(0, np.log(prob_wc / (prob_word * prob_class)).round(3))

            # give PMI score of 0 if word occurs less than x times in the dataset
            else:
                pmi[label][word] = 0.0
    return pmi

pmi_scores = features(X_train_hypothesis, y_train)

# save PMI scores as json files (change filename for each dataset)
output_file_path = 'HO_pmi_scores_dataset.json'
with open(output_file_path, 'w') as output_file:
    json.dump(pmi_scores, output_file)
