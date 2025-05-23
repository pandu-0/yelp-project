#Import Statements
import pandas as pd
from collections import Counter
import nltk
nltk.download('punkt')
from nltk import word_tokenize
import numpy as np
from scipy import stats

philly_reviews = pd.read_json('copy_philly_reviews.json', lines=True)

philly_reviews

#Getting mean reviews by open status
philly_reviews[['is_open','review_stars']].groupby('is_open').mean()

#Getting correlation between open status and reviews
philly_reviews[['is_open','review_stars']].corr()

#Getting p value for t test - p value is zero indicating the sample set was too big
n = len(philly_reviews)

r = 0.054492

t_stat = r * np.sqrt((n - 2) / (1 - r**2))

df = n - 2

p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df))

print(f"t-statistic: {t_stat}")
print(f"p-value: {p_value:.10e}")

#Getting p value for t test with smaller sample
sample_df = philly_reviews[['is_open', 'review_stars']].sample(n=1000, random_state=42)

r = sample_df.corr().loc['is_open', 'review_stars']

n = len(sample_df)
t_stat = r * np.sqrt((n - 2) / (1 - r**2))
df = n - 2
p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df))

print(f"Sample size: {n}")
print(f"Correlation (r): {r}")
print(f"t-statistic: {t_stat}")
print(f"p-value: {p_value:.10f}")

nltk.download('punkt_tab')

#Tokenizing review example
first_review = philly_reviews.iloc[0]['review']
tokens = word_tokenize(first_review)

tokens

nltk.download('averaged_perceptron_tagger_eng')

#Random sampling since dataset is too large to process
philly_reviews_sampled = philly_reviews.sample(n=10000, random_state=42)

#Tokenizing and Part of Speech tagging all the reviews in the sampled dataset and counting the frequency of each word
all_word_pos = []

for review in philly_reviews_sampled['review']:
    tokens = word_tokenize(review.lower())
    pos_tags = nltk.pos_tag(tokens)
    all_word_pos.extend(pos_tags)


word_pos_freq = Counter(all_word_pos)

word_pos_freq.most_common()

df = pd.DataFrame(word_pos_freq.most_common(), columns=['word', 'count'])
df

df[['word', 'pos']] = pd.DataFrame(df['word'].tolist(), index=df.index)

#Filtering the above list of words to only include adjectives
adjectives_df = df[df['pos'].isin(['JJ', 'JJR', 'JJS'])]

adjectives_df

#Filtering out all the adjectives that occured less than 5 times to reduce computation time
adjectives_df = adjectives_df[adjectives_df['count']>5]

#Creating a matrix to store the presence of a adjective by the rating
adjective_list = adjectives_df['word'].tolist()

features = []

for review, rating in zip(philly_reviews_sampled['review'], philly_reviews_sampled['review_stars']):
    tokens = word_tokenize(review.lower())
    adjectives_in_review = [word for word in tokens if word in adjective_list]

    feature_row = [1 if adj in adjectives_in_review else 0 for adj in adjective_list]

    features.append(feature_row + [rating])

adjective_matrix_df = pd.DataFrame(features, columns=adjective_list + ['rating'])

print(adjective_matrix_df.head())

#Using the above matrix and caluculating the correlation between the presence of the adjective and the rating and printing the 20 highest correlated values
correlation_matrix = adjective_matrix_df.corr()

adjective_rating_correlation = correlation_matrix['rating'].sort_values(ascending=False)
adjective_rating_correlation_1 = adjective_rating_correlation.drop_duplicates()

print(adjective_rating_correlation.head(20))

#printing the 20 lowest correlated values
print(adjective_rating_correlation.tail(20))