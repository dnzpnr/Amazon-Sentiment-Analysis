import pandas as pd
import numpy as np
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = pd.read_csv('amazon_alexa.tsv', sep= '\t')
reviews_df = df.copy()
reviews_df.head()

reviews_df.info()

reviews_df.hist(bins = 30, figsize = (13,5), color = 'r');

reviews_df['length'] = reviews_df['verified_reviews'].apply(len)
reviews_df.head()

reviews_df['length'].plot(bins=30, kind='hist');

reviews_df.length.describe()

positive = reviews_df[reviews_df['feedback']==1]
negative = reviews_df[reviews_df['feedback']==0]
positive.head()

negative.head()

negative.head()

sns.countplot(x = 'rating', data = positive)

sns.countplot(x = 'rating', data = negative)

variation_table = reviews_df.groupby('variation')['feedback'].value_counts()
variation_table

sentences = reviews_df['verified_reviews'].to_list()
sentences[:5]

sentences_as_one_string = ' '.join(sentences)
sentences_as_one_string[:100]

from wordcloud import WordCloud
fig, ax = plt.subplots(figsize=(20,20))
plt.imshow(WordCloud().generate(sentences_as_one_string))
ax.axis('off')

negative_list = negative['verified_reviews'].tolist()
negative_sentences_as_one_string = " ".join(negative_list)
plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(negative_sentences_as_one_string))

import string
string.punctuation

reviews_df['verified_reviews'] = reviews_df['verified_reviews'].apply(lambda x: [char for char in x if char not in string.punctuation])
reviews_df['verified_reviews'][:50]

reviews_df['verified_reviews'] = reviews_df['verified_reviews'].apply(lambda x: ''.join(x))
reviews_df['verified_reviews'][:50]

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stopwords.words('english')

reviews_df['verified_reviews'] = reviews_df['verified_reviews'].apply(lambda x: [word for word in x.split() if word.lower() not in stopwords.words('english')])
reviews_df['verified_reviews'][:50]

reviews_df['verified_reviews'] = reviews_df['verified_reviews'].apply(lambda x: ' '.join(x))
reviews_df['verified_reviews'][:50]

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(reviews_df['verified_reviews'])
print(x.toarray())

print(vectorizer.get_feature_names()[:100])

x.shape

reviews = pd.DataFrame(x.toarray())
reviews.head()

reviews_df = pd.concat([reviews_df, reviews], axis=1)
reviews_df.head()

reviews_df['date'].value_counts()

reviews_df.groupby('date')['feedback'].value_counts()

reviews_df.groupby('variation')['feedback'].value_counts()


reviews_df = pd.get_dummies(reviews_df, columns= ['variation'],drop_first=True)
reviews_df.head()

reviews_df.groupby('length')['feedback'].value_counts()
sns.scatterplot(data=reviews_df, x="length", y= reviews_df.groupby('length')['feedback'].count(), hue= "feedback")

reviews_df.drop(['date','verified_reviews'],axis=1, inplace=True)
reviews_df.head()

sns.heatmap(reviews_df.isnull(),cbar=False)

X = reviews_df.drop(['feedback'],axis=1)
y = reviews_df['feedback']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix
y_pred = NB_classifier.predict(X_train)
confusion_matrix(y_train, y_pred)

y_pred_test = NB_classifier.predict(X_test)
print(classification_report(y_test, y_pred_test))

from sklearn.metrics import accuracy_score
accuracy_score(y_pred_test, y_test)