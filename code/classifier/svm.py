# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import pickle


class SVMClassifierModel:
    def __init__(self, df):
        self.df = df
        self.vect = None
        self.model = None

    def load_vectorizer(self, path):
        # Load the pickled vectorizer
        with open(path, "rb") as file:
            self.vect = pickle.load(file)

    def preprocess_data(self):
        # Filter the DataFrame for 'Democrat' and 'Republican' parties
        self.df = self.df[(self.df['party'] == 'Democrat') | (self.df['party'] == 'Republican')]

    def split_data(self):
        # Split data into training and testing sets
        return train_test_split(
            self.df['transcript'], self.df['party'],
            test_size=0.2, random_state=1234, stratify=self.df['party'])

    def transform_data(self, x_train, x_test):
        # Transform text data into a matrix of token counts
        return self.vect.transform(x_train), self.vect.transform(x_test)

    def train_model(self, x_train, y_train):
        # Initialize and train the SVM Classifier
        self.model = SVC(kernel='linear', probability=True, random_state=1234)
        self.model.fit(x_train, y_train)

    def evaluate_model(self, x_test, y_test):
        # Evaluate the model
        y_predict_svm = self.model.predict(x_test)
        accuracy = accuracy_score(y_test, y_predict_svm)
        print("SVM Accuracy Score:", accuracy)

        # Compute AUC
        y_prob = self.model.predict_proba(x_test)[:, 1]  # Ensure the positive class is 'Democrat'
        auc = roc_auc_score(y_test, y_prob)
        print("SVM AUC:", auc)


# Usage example
# svm_model = SVMClassifierModel(content_df)
# svm_model.load_vectorizer('path_to_vectorizer.pkl')
# svm_model.preprocess_data()
# x_train, x_test, y_train, y_test = svm_model.split_data()
# x_train_transformed, x_test_transformed = svm_model.transform_data(x_train, x_test)
# svm_model.train_model(x_train_transformed, y_train)
# svm_model.evaluate_model(x_test_transformed, y_test)