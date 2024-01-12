# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
import pickle


class LogisticRegressionModel:
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
        x_train, x_test, y_train, y_test = train_test_split(
            self.df['transcript'], self.df['party'],
            test_size=0.2, random_state=1234, stratify=self.df['party'])
        return x_train, x_test, y_train, y_test

    def transform_data(self, x_train, x_test):
        # Transform text data into a matrix of token counts
        return self.vect.transform(x_train), self.vect.transform(x_test)

    def train_model(self, x_train, y_train):
        # Initialize and train the Logistic Regression Model
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(x_train, y_train)

    def evaluate_model(self, x_test, y_test):
        # Evaluate the model
        accuracy = self.model.score(x_test, y_test)
        print("Logistic Regression Accuracy Score:", accuracy)

        # Compute AUC
        y_prob = self.model.predict_proba(x_test)[:, 1]  # Ensure the positive class is 'Democrat'
        auc = roc_auc_score(y_test, y_prob)
        print("Logistic Regression AUC:", auc)
