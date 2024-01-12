# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series


class PoliticalLeaningsPredictor:
    def __init__(self, model, vectorizer):
        self.model = model
        self.vect = vectorizer

    def predict(self, text):
        # Transform the text using the vectorizer and make a prediction
        x_test = self.vect.transform(Series(text))
        return self.model.predict(x_test)[0]
    
    def predict_for_president(self, df, president_name):
        # Apply the predict method to each transcript of the specified president
        return print(df[df['name'] == president_name]['transcript'].apply(self.predict))

    def plot_party_distribution(self, df, president_name):
        """
        Plots a pie chart of party distribution for a given president's transcripts.
        """
        predictions = df[df['name'] == president_name]['transcript'].apply(self.predict)

        # Counting Democrat and Republican predictions
        dem_count = sum(predictions == 'Democrat')
        rep_count = sum(predictions == 'Republican')

        # Set figure size and other parameters
        plt.rcParams['figure.figsize'] = [5, 5]
        ratios, labels, colors, explode = self._get_plot_params(dem_count, rep_count)

        plt.pie(ratios, labels=labels, autopct='%.0f%%', colors=colors,
                textprops={'fontsize': 15, 'weight': 'bold'}, explode=explode,
                startangle=-70, wedgeprops={'width': 1, "linewidth": 2, "edgecolor": 'black'}, shadow=True)

        title_color = 'royalblue' if dem_count > rep_count else 'firebrick'
        plt.title(president_name, size=20, fontdict={'fontweight': 'bold'}, 
                  bbox={'facecolor': title_color, 'pad': 5, 'alpha': 0.7}, loc='center')

    def _get_plot_params(self, dem_count, rep_count):
        # Determine ratios, labels, colors, and explode values based on counts
        if dem_count > 0 and rep_count > 0:
            return ([dem_count, rep_count], ['Democrat', 'Republican'], ['royalblue', 'firebrick'], (0.1, 0.0))
        elif rep_count == 0:
            return ([dem_count], ['Democrat'], ['royalblue'], (0.1,))
        elif dem_count == 0:
            return ([rep_count], ['Republican'], ['firebrick'], (0.1,))