# -*- coding: utf-8 -*-


# libraries 
import pandas as pd
import time

# custom libraries 
from scraping import url_scraping, content_scraping
from preprocessing import transcript_preprocessing, party_labeling
from classifier import naive_bayes, xgboost, lightgbm, logisticregression, svm, randomforest
from predict import predict
from visualization import wordclouds, hbargraph, partypiechart, timeline, custompiechart
from utilities import *


def run_model(model_class, content_df, vectorizer_path, has_valid_data=False, **model_kwargs):
    # Initialize the model with additional arguments
    model = model_class(content_df, **model_kwargs)
    model.load_vectorizer(vectorizer_path)
    model.preprocess_data()

    if has_valid_data:
        x_train, x_test, x_valid, y_train, y_test, y_valid = model.split_data()
        x_train_transformed, x_test_transformed, x_valid_transformed = model.transform_data(x_train, x_test, x_valid)
        model.train_model(x_train_transformed, y_train, x_valid_transformed, y_valid)
    else:
        x_train, x_test, y_train, y_test = model.split_data()
        x_train_transformed, x_test_transformed = model.transform_data(x_train, x_test)
        model.train_model(x_train_transformed, y_train)

    model.evaluate_model(x_test_transformed, y_test)

    return model


def main():
    # Scrape URLs
    website = 'https://millercenter.org/the-presidency/presidential-speeches'
    url_scraper = url_scraping.URLScraper(website)
    url_scraper.load_page()
    url_scraper.scroll_page() 
    url_scraper.extract_speech_urls()
    speech_urls = url_scraper.speech_urls 
    url_scraper.close_driver()
    print('Number of Speech Urls :', len(speech_urls))
    
    # Dump & Load list of speech URLs
    dump('c:/Python Projects/US_Presidential_Speech/data/speech_urls.txt', speech_urls)
    speech_urls = load('c:/Python Projects/US_Presidential_Speech/data/speech_urls.txt')
    
    # Scrape contents from URLs
    content_df = pd.DataFrame()
    content_scraper = content_scraping.ContentScraper()
    for speech_url in speech_urls:
        data = content_scraper.scrape_speech_content(speech_url)
        content_df = pd.concat([content_df, data], ignore_index=True)
        time.sleep(2)
    print(content_df.info())
    print(content_df)
    
    # Transcript preprocessing
    transcript_preprocessor = transcript_preprocessing.TranscriptPreprocessor()
    content_df = transcript_preprocessor.preprocess_transcripts(content_df)
    print(content_df.info())
    print(content_df)
    
    # Party Affilation Labeling
    party_labeler = party_labeling.PartyLabeler()
    content_df = party_labeler.add_party_affiliation(content_df)
    print(content_df.info())
    print(content_df)
    
    # Save & load the preprocessed content dataframe as a csv file
    content_df.to_csv('c:/Python Projects/US_Presidential_Speech/data/us_presidential_speech.csv', index=False)
    content_df = pd.read_csv('c:/Python Projects/US_Presidential_Speech/data/us_presidential_speech.csv')
    
    # Naive Bayes Classifier 
    nbc = naive_bayes.NaiveBayesClassifier(content_df)
    nbc.preprocess_data()
    x_train, x_test, y_train, y_test = nbc.split_data()
    nbc.vectorize_data(x_train)
    vect = nbc.vect
    x_train_transformed = nbc.transform_data(x_train)
    x_test_transformed = nbc.transform_data(x_test)
    nbc.train_model(x_train_transformed, y_train)
    nbm = nbc.model
    nbc.evaluate_model(x_test_transformed, y_test)
    
    # Dump the vectorizer and the trained model from Naive Bayes Classifer
    dump('c:/Python Projects/US_Presidential_Speech/data/vect.pkl', vect)
    dump('c:/Python Projects/US_Presidential_Speech/data/nbmodel.pkl', nbm)
    
    # Using the same vect from Naive Bayes
    vectorizer_path = 'c:/Python Projects/US_Presidential_Speech/data/vect.pkl'

    # XGBoost Classifier
    run_model(xgboost.XGBClassifierModel, content_df, vectorizer_path, has_valid_data=True)
    
    # lightgbm Classifer
    run_model(lightgbm.LGBMClassifierModel, content_df, vectorizer_path, has_valid_data=True)
    
    # Logistic Regression
    run_model(logisticregression.LogisticRegressionModel, content_df, vectorizer_path)
    
    # SVM (Support Vector Machine)
    run_model(svm.SVMClassifierModel, content_df, vectorizer_path)
    
    # Random Forest with 'entropy' criterion
    run_model(randomforest.RandomForestClassifierModel, content_df, vectorizer_path, criterion='entropy')
    
    # Random Forest with 'gini' criterion
    run_model(randomforest.RandomForestClassifierModel, content_df, vectorizer_path, criterion='gini')
    
    # Load the vectorizer and the trained model from Naive Bayes Classifer
    vect = load('c:/Python Projects/US_Presidential_Speech/data/vect.pkl')
    nbm = load('c:/Python Projects/US_Presidential_Speech/data/nbmodel.pkl')
    
    # Predict the Political Leanings of U.S. Presidents from Other Political Parties by Classifying Their Speeches into Democrat or Republican
    # Example using Andrew Johnson
    pred = predict.PoliticalLeaningsPredictor(nbm, vect)
    pred.predict_for_president(content_df, 'Andrew Johnson')
    pred.plot_party_distribution(content_df, 'Andrew Johnson')
    
    # Word Cloud (Total & All)
    wordcloud = wordclouds.PoliticalWordCloudGenerator(content_df)
    wordcloud.preprocess_data()
    wordcloud.generate_word_cloud(party=None, freq_threshold=0)
    
    # Horizontal Bar Graph (Total & >= 5000)
    bar = hbargraph.PoliticalBarGraphGenerator(content_df)
    bar.preprocess_data()
    bar.generate_bar_graph(party=None, freq_threshold=5000)
    
    # Political Party : # of speech pie chart
    pie = partypiechart.PartyDistributionPieChart(content_df)
    pie.generate_pie_chart()

    # Graph of U.S. President's Political Party Affilliation by Year
    yearly_timeline = timeline.PartyAffiliationTimeline(content_df)
    yearly_timeline.preprocess_data()
    yearly_timeline.plot_timeline()

    # Custom Pie Charts
    train_test_pie_chart = custompiechart.CustomPieChart(
        ratio=[423, 467],
        labels=['Republican', 'Democrat'],
        colors=['firebrick', 'royalblue'],
        explode=(0, 0),  
        startangle=90
        )
    train_test_pie_chart.plot()

    model_accuracy_pie_chart = custompiechart.CustomPieChart(
        ratio=[21, 157],
        labels=['Fail', 'Success'],
        colors=['red', 'mediumseagreen'],
        explode=(0.1, 0),  # Only explode the first slice ('Fail')
        startangle=0
        )
    model_accuracy_pie_chart.plot()
    
if __name__ == '__main__':
    main()

