# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from .textanalyzer import PoliticalTextAnalyzer

class PoliticalBarGraphGenerator(PoliticalTextAnalyzer):
    def __init__(self, df):
        super().__init__(df)

    def generate_bar_graph(self, party=None, freq_threshold=0):
        if party not in self.tokenized_data:
            self.calculate_frequencies(party)
        freq_data = self.tokenized_data[party]
        freq_data = freq_data[freq_data > freq_threshold]

        fig, ax = plt.subplots(figsize=(7, 7))
        p = ax.barh(freq_data.index, freq_data, color='coral' if party != 'Republican' else 'firebrick', align='center')
        ax.set_title(party if party else "Total", weight='bold', fontsize=15)
        ax.invert_yaxis()
        ax.set_xlim(0, max(freq_data) * 1.1)
        ax.bar_label(p, padding=3, label_type='edge', fontsize=10, fontweight='bold', color='black')
        plt.show()

# Usage example in another script
# from political_bargraph_generator import PoliticalBarGraphGenerator
# bar_graph_generator = PoliticalBarGraphGenerator(content_df)
# bar_graph_generator.preprocess_data()
# bar_graph_generator.generate_bar_graph(party='Democrat', threshold=2000)
