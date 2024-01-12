# -*- coding: utf-8 -*-

import lightgbm as lgb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import pandas as pd
import pickle


class LGBMClassifierModel:
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
            test_size=0.1, random_state=1234, stratify=self.df['party_binary'])

        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train, y_train, test_size=0.2, random_state=7777, stratify=y_train)

        return x_train, x_test, x_valid, y_train, y_test, y_valid

    def transform_data(self, x_train, x_test, x_valid):
        # Transform text data into a matrix of token counts
        x_train_transformed = self.vect.transform(x_train).astype(float)
        x_test_transformed = self.vect.transform(x_test).astype(float)
        x_valid_transformed = self.vect.transform(x_valid).astype(float)
        return x_train_transformed, x_test_transformed, x_valid_transformed

    def train_model(self, x_train, y_train, x_valid, y_valid):
        # Convert labels to float
        y_train = y_train.astype(float)
        y_valid = y_valid.astype(float)
        
        # Initialize and train the LightGBM Classifier
        evals = [(x_train, y_train), (x_valid, y_valid)]
        self.model = lgb.LGBMClassifier(n_estimators=1000, early_stopping_rounds=100)
        self.model.fit(x_train, y_train, eval_set=evals, eval_metric='logloss',
                  callbacks=[lgb.log_evaluation(period=0)])  # Set period=0 for no output

    def evaluate_model(self, x_test, y_test):
        # Evaluate the model
        accuracy = self.model.score(x_test, y_test)
        print("LightGBM Accuracy Score:", accuracy)

        # Compute AUC
        y_prob = self.model.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        print("LightGBM AUC:", auc)