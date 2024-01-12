# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import pandas as pd
import matplotlib.pyplot as plt


class NaiveBayesClassifier:
    def __init__(self, df):
        self.df = df
        self.vect = None
        self.model = None

    def preprocess_data(self):
        # Filter the DataFrame for 'Democrat' and 'Republican' parties
        self.df = self.df[(self.df['party'] == 'Democrat') | (self.df['party'] == 'Republican')]

    def split_data(self):
        # Split data into training and testing sets
        return train_test_split(self.df['transcript'],
                                self.df['party'],
                                test_size=0.2,
                                random_state=1234,
                                stratify=self.df['party'])

    def vectorize_data(self, x_train):
        # Initialize and fit CountVectorizer
        stopwords_list = stopwords.words('english')
        self.vect = CountVectorizer(stop_words=stopwords_list, ngram_range=(1, 3))
        self.vect.fit(x_train)
        

    def transform_data(self, x_data):
        # Transform text data into a matrix of token counts
        return self.vect.transform(x_data)

    def train_model(self, x_train, y_train):
        # Initialize and train the Naive Bayes Classifier
        self.model = MultinomialNB()
        self.model.fit(x_train, y_train)

    def evaluate_model(self, x_test, y_test):
        # Evaluate the model
        accuracy = self.model.score(x_test, y_test)
        print("Multinomial Naive Bayes Accuracy Score:", accuracy, '\n')
        
        # Confusion matrix (crosstab)
        y_predict = self.model.predict(x_test)
        print("Confusion Matrix:\n", pd.crosstab(y_test, y_predict), '\n')
 
        # Classification report
        print("Classification Report:\n", classification_report(y_test, y_predict), '\n')

        # Compute AUC
        y_prob = self.model.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        print("Multinomial Naive Bayes AUC:", auc)

        # Plot the ROC curve
        y_test_binary = (y_test == 'Republican').astype(int)  # Convert string labels to binary format
        fpr, tpr, _ = roc_curve(y_test_binary, y_prob)
        plt.plot(fpr, tpr, label=f"Multinomial Naive Bayes (AUC = {auc:.2f})")
        plt.plot([0, 1], [0, 1], 'r--', label='Random')
        plt.xlabel('False Positive Rate (1 − Specificity)')
        plt.ylabel('True Positive Rate (Recall/Sensitivity)')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()

