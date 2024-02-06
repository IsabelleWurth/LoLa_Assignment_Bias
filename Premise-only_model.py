# import libraries
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
import json
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Replace train and test datasets with the actual file paths of your JSON datasets
input_file_path_PMI = 'your_pmi_scores.json'
input_file_path_train = 'your_train_data.json'
input_file_path_test = 'your_test_data.json'


with open(input_file_path_PMI, 'r') as json_file:
    pmi_scores = json.load(json_file)

with open(input_file_path_train, 'r') as json_file:
    train_data = json.load(json_file)

with open(input_file_path_test, 'r') as json_file:
    test_data = json.load(json_file)

# preprocess data
X_train_premise = [my_dict['p'] for my_dict in train_data]
X_test_premise = [my_dict['p'] for my_dict in test_data]
y_train = [my_dict['g'] for my_dict in train_data]
y_test = [my_dict['g'] for my_dict in test_data]

X_train_premise = X_train_premise[:20000]
X_test_premise = X_test_premise[:5000]
y_train = y_train[:20000]
y_test = y_test[:5000]
# preprocess PMI scores for the model
def extract_features(document, pmi_scores, label):
    features = {word: pmi_scores[label].get(word, 0.0) for word in document.split()}
    return features

X_train_features = [extract_features(doc, pmi_scores, label) for doc, label in zip(X_train_premise, y_train)]
X_test_features = [extract_features(doc, pmi_scores, label) for doc, label in zip(X_test_premise, y_test)]

# find top words that correlate with each label
def top_words(pmi_scores, top_n=5):
    top_words = {}
    for label in pmi_scores.keys():
        top_words[label] = sorted(pmi_scores[label].items(), key=lambda x: x[1], reverse=True)[:top_n]
    return top_words

# print top words
top_list = top_words(pmi_scores)
for label, words in top_list.items():
    print(f"Top words for label '{label}':")
    for word, score in words:
        print(f"{word}: {score}")
    print("\n")

# make features into vectors for classifying
def modelling(X_train_features, X_test_features, y_train, y_test):
    vectorizer = DictVectorizer(sparse=False)
    X_train_numeric = vectorizer.fit_transform(X_train_features)
    X_test_numeric = vectorizer.transform(X_test_features)

    # Naive bayes with smooth 100 to deal with 0 probability features 
    clf = MultinomialNB(alpha = 100)
    clf.fit(X_train_numeric, y_train)

    #Make predictions
    predictions = clf.predict(X_test_numeric)

    # Evaluate the model 
    accuracy = metrics.accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions, labels=['entailment', 'neutral', 'contradiction'])

    #print("Value counts in y_test:\n", pd.Series(y_test).value_counts())
    #print("Value counts in predictions:\n", pd.Series(predictions).value_counts())

    # Plot the confusion matrix
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['entailment', 'neutral', 'contradiction'], yticklabels=['entailment', 'neutral', 'contradiction'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Display other metrics
    print("Accuracy:", round((accuracy_score(y_test, predictions) * 100),2))
    print("Classification Report:\n", classification_report(y_test, predictions))
    return accuracy

acc = modelling(X_train_features, X_test_features, y_train, y_test)
print(acc)