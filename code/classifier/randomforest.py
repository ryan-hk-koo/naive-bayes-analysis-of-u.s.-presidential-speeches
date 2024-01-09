import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from nltk.corpus import stopwords
import pickle


class RandomForestClassifierModel:
    def __init__(self, df, criterion='entropy'):
        self.df = df
        self.criterion = criterion
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
        # Initialize and train the Random Forest Classifier
        self.model = RandomForestClassifier(criterion=self.criterion, n_estimators=500, oob_score=True, random_state=1234)
        self.model.fit(x_train, y_train)

    def evaluate_model(self, x_test, y_test):
        # Evaluate the model
        accuracy = self.model.score(x_test, y_test)
        print(f"Random Forest ({self.criterion} criterion) Accuracy Score:", accuracy)

         # Compute AUC
        y_prob = self.model.predict_proba(x_test)[:, 1]  # Ensure the positive class is 'Democrat'
        auc = roc_auc_score(y_test, y_prob)
        print(f"Random Forest ({self.criterion} criterion) AUC:", auc)

# Usage example
# rf_model_gini = RandomForestClassifierModel(content_df, criterion='gini')
# rf_model_entropy = RandomForestClassifierModel(content_df, criterion='entropy')
# [Load vectorizer, preprocess data, split, transform, train, and evaluate for both models]