# -*- coding: utf-8 -*-

from pandas import Series
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords


class PoliticalTextAnalyzer:
    def __init__(self, df):
        self.df = df
        self.stopwords = set(stopwords.words('english'))
        self.stopwords.update(['has', 'is', 'are', 'not', 'have', 'be', 'were', 'been', 'n\'t', 
                               'mr.', '\'s', '\'re', '\'ve', 'said', 'much', 'president', 'united', 
                               'states', 'also', 'america', 'american', 'years'])
        self.tokenized_data = {}

    def preprocess_data(self):
        def nltk_pos(arg):
            tagged_list = pos_tag(word_tokenize(arg))
            word_tagg = [word[0] for word in tagged_list if word[1] in ['NN', 'NNP', 'NNS', 'NNPS', 
                                                                       'VBP', 'VBZ', 'VB', 'VBD', 'VBN', 
                                                                       'JJ', 'JJR', 'JJS', 
                                                                       'RB', 'RBR', 'RBS']]
            return word_tagg

        self.df['token'] = self.df['transcript'].apply(nltk_pos)

    def calculate_frequencies(self, party=None):
        tokens = self.df['token'] if party is None else self.df[self.df['party'] == party]['token']
        tokens = [j for i in tokens for j in i if len(j) >= 2 and j not in self.stopwords]
        self.tokenized_data[party] = Series(tokens).value_counts()