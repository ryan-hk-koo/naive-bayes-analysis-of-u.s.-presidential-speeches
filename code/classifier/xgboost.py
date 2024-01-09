# -*- coding: utf-8 -*-

import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import pandas as pd
import pickle
from nltk.corpus import stopwords


class XGBClassifierModel:
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
        filtered_df = self.df[(self.df['party'] == 'Democrat') | (self.df['party'] == 'Republican')].copy()
        # Convert 'party' to binary: 1 for 'Democrat' and 0 for 'Republican'
        filtered_df['party_binary'] = np.where(filtered_df['party'] == 'Democrat', 1, 0)
        self.df = filtered_df

    def split_data(self):
        # Split data into training, testing, and validation sets
        x_train, x_test, y_train, y_test = train_test_split(
            self.df['transcript'], self.df['party_binary'],
            test_size=0.2, random_state=1234, stratify=self.df['party_binary'])

        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train, y_train, test_size=0.1, random_state=7777, stratify=y_train)

        return x_train, x_test, x_valid, y_train, y_test, y_valid

    def transform_data(self, x_train, x_test, x_valid):
        # Transform text data into a matrix of token counts
        return self.vect.transform(x_train), self.vect.transform(x_test), self.vect.transform(x_valid)

    def train_model(self, x_train, y_train, x_valid, y_valid):
        # Initialize and train the XGBoost Classifier
        evals = [(x_train, y_train), (x_valid, y_valid)]
        self.model = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, eval_metric='logloss', early_stopping_rounds=50)
        self.model.fit(x_train, y_train, eval_set=evals, verbose=False)

    def evaluate_model(self, x_test, y_test):
        # Evaluate the model
        accuracy = self.model.score(x_test, y_test)
        print("XGBoost Accuracy Score:", accuracy)

        # Compute AUC
        y_prob = self.model.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        print("XGBoost AUC:", auc)