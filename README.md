![image](https://github.com/ryan-hk-koo/naive_bayes_analysis_of_u.s._presidential_speeches/assets/143580734/5cb96d6d-f8f8-4d63-9aa5-0dbe1316fd39)

# Purpose
- Examine whether there is a discernible difference between Republican and Democrat presidential speeches using the Naïve Bayes Classifier
- Determine whether speeches by U.S. presidents from other political parties aligned more closely with Democratic or Republican rhetorical patterns using the Naïve Bayes Classifier developed above
- Visualize key rhetorical elements in Democratic and Republican speeches using word clouds and bar graphs

# Dataset
![image](https://github.com/ryan-hk-koo/naive_bayes_analysis_of_u.s._presidential_speeches/assets/143580734/43c35d8d-cbf1-431f-a882-8a06b0213f08)
<br>
<br>
Source : The Miller Center (University of Virginia) https://millercenter.org/the-presidency/presidential-speeches  


# Method
<br>

## Classifier Model Selection
![image](https://github.com/ryan-hk-koo/naive_bayes_analysis_of_u.s._presidential_speeches/assets/143580734/47461eb9-6333-4462-b8f9-03a9d5c9554a)
- Tested seven different classifier models including Multinomial Naive Bayes, XGBoost, LightGBM, Logistic Regression, SVM, and two variations of  Random Forest
- For U.S. presidential speech text data, the Multinomial Naive Bayes classifier outperformed all the other models by achieving an AUC of 0.92 and an accuracy score of 88%
- Hence, Multinomial Naive Bayes was selected as the classifier model for this project

<br>

![image](https://github.com/ryan-hk-koo/naive_bayes_analysis_of_u.s._presidential_speeches/assets/143580734/8e8ce796-f3ab-4e29-9d10-c9706c9f997b)
- Used Naïve Bayes classifier because of its proven efficiency in text classification and its unique ability to provide nuanced probabilistic insights
- Additionally, its scalability ensured seamless processing of the extensive speech corpus of U.S. presidential speeches

<br>

![image](https://github.com/ryan-hk-koo/naive_bayes_analysis_of_u.s._presidential_speeches/assets/143580734/9ea11e0c-a7f0-4bdb-a8d7-a495e35eff9a)
- ROC curve of the Multinomial Naive Bayes Classifier 

# Performance Analysis
![image](https://github.com/ryan-hk-koo/naive_bayes_analysis_of_u.s._presidential_speeches/assets/143580734/899d8a4a-ef23-400b-84ee-63faa3e15aa3)

# Analysis

## Classifying Speeches of U.S. Presidents from Other Political Parties

![image](https://github.com/ryan-hk-koo/naive_bayes_analysis_of_u.s._presidential_speeches/assets/143580734/559462ea-b705-4b3a-9a82-9f2f7146b7da)

<br>
<br>

![image](https://github.com/ryan-hk-koo/naive_bayes_analysis_of_u.s._presidential_speeches/assets/143580734/361be8a8-a347-4ae1-99e6-7f80a0ada9ee)

<br>
<br>

![image](https://github.com/ryan-hk-koo/naive_bayes_analysis_of_u.s._presidential_speeches/assets/143580734/e4eab760-3c52-44c9-aca4-6302e77a2e04)

<br>
<br>

![image](https://github.com/ryan-hk-koo/naive_bayes_analysis_of_u.s._presidential_speeches/assets/143580734/0a60a55f-0b5d-47e9-9ac2-9f26e68be323)

<br>
<br>

![image](https://github.com/ryan-hk-koo/naive_bayes_analysis_of_u.s._presidential_speeches/assets/143580734/a7f475c0-067f-4c5f-b0e8-afbb11e0744a)

<br>

## Word Clouds & Bar Graphs

<br>

![image](https://github.com/ryan-hk-koo/naive_bayes_analysis_of_u.s._presidential_speeches/assets/143580734/65c5186c-be71-4c07-bbc8-d9d47da6e6b8)

<br>
<br>

![image](https://github.com/ryan-hk-koo/naive_bayes_analysis_of_u.s._presidential_speeches/assets/143580734/25737a07-e34f-44a6-85cb-d246887632b6)

<br>
<br>

![image](https://github.com/ryan-hk-koo/naive_bayes_analysis_of_u.s._presidential_speeches/assets/143580734/61bdaa2d-1dc6-4155-9c31-c2193f1f4b24)

Regarding words occuring more than 2,000 times : 
- Democratic presidential speeches uniquely featured "think" and "power"
- Republican presidential speeches uniquely featured "law" and "work"
<br>

![image](https://github.com/ryan-hk-koo/naive_bayes_analysis_of_u.s._presidential_speeches/assets/143580734/a70438bd-c5fe-4885-a781-44b9eff9f05c)

<br>
<br>

![image](https://github.com/ryan-hk-koo/naive_bayes_analysis_of_u.s._presidential_speeches/assets/143580734/03aed27f-02cc-4f82-ab6e-72a85930e02f)

<br>

# Conclusion
- The Naïve Bayes Classifier achieved an accuracy of 88%, confirming a discernible difference between Republican and Democrat presidential speeches
- The Naïve Bayes Classifier's categorization of speeches from U.S. presidents of other political parties into Democratic and Republican categories aligned with historical facts regarding their parties, further attesting to the model's accuracy
- Visualizing key rhetorical elements highlighted distinct words in Democratic and Republican speeches, yet many were strikingly similar. This underscores the importance of employing the n-gram technique for a more nuanced analysis
- The trained Naïve Bayes Classifier can potentially be used to assess the political leanings of future U.S. presidential candidates by classifying their speeches as either Democratic or Republican






