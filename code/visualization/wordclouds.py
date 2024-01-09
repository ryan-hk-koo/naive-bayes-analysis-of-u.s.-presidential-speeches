# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from .textanalyzer import PoliticalTextAnalyzer

class PoliticalWordCloudGenerator(PoliticalTextAnalyzer):
    def __init__(self, df):
        super().__init__(df)

    def generate_word_cloud(self, party=None, freq_threshold=0, mask_shape='circle'):
        if party not in self.tokenized_data:
            self.calculate_frequencies(party)

        freq_data = self.tokenized_data[party]
        freq_data = freq_data[freq_data > freq_threshold]

        mask = None
        if mask_shape == 'circle':
            x, y = np.ogrid[:300, :300]
            mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
            mask = 255 * mask.astype(int)

        wordcloud = WordCloud(width=1000, height=1000, background_color='black',
                              colormap='RdBu', mask=mask).generate_from_frequencies(freq_data)

        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')
        plt.show()


# Example usage
# generator = PoliticalWordCloudGenerator(content_df)
# generator.preprocess_data()
# generator.generate_word_cloud(party='Republican', freq_threshold=2000)