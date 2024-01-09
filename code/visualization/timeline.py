# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class PartyAffiliationTimeline:
    def __init__(self, df):
        self.df = df

    def preprocess_data(self):
        self.df['date'] = pd.to_datetime(self.df['date'], format='%B %d, %Y')
        self.df['year'] = self.df['date'].dt.year
        self.df['number'] = None

        self.df.loc[self.df['party'] == 'Democrat', 'number'] = -1
        self.df.loc[self.df['party'] == 'Republican', 'number'] = 1

    def plot_timeline(self):
        s = self.df['year']
        t = self.df['number']
        fig, ax = plt.subplots()

        ax.plot(t, s, color='black')
        ax.axvline(0, color='black')

        ax.fill_betweenx(s, 1, where=t > 0, facecolor='firebrick', alpha=1)
        ax.fill_betweenx(s, -1, where=t < 0, facecolor='royalblue', alpha=1)

        ax.invert_yaxis()
        plt.xlabel("Democrat (Blue) vs. Republican (Red)", fontdict={'fontweight': 'bold'})
        plt.ylabel("Year", fontdict={'fontweight': 'bold'})
        ax.axes.get_xaxis().set_ticks([])
        plt.show()

# Usage
# Assuming 'df' is your DataFrame with a 'date' and 'party' column
# timeline = PartyAffiliationTimeline(df)
# timeline.preprocess_data()
# timeline.plot_timeline()
