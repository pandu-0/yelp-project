# Import libraries
from sklearn.decomposition import NMF
import pandas as pd
from tqdm.notebook import tqdm
from bertopic import BERTopic
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer


# REPLACE FILE PATH WITH YOUR OWN
philly_restaurant_reviews = pd.read_json('/yelp-dataset/philly_restaurants_reviews_with_sentiment_complete.json', lines=True)

# REPLACE FILE PATH WITH YOUR OWN
philly_balanced_clean = pd.read_json('/yelp-dataset/philly_balanced_with_sentiment_clean.json', lines=True)

documents = philly_restaurant_reviews['review'].tolist()

# --- Step 1: Convert the reviews to TF-IDF vectors ---
vectorizer = TfidfVectorizer(
    stop_words='english',   # remove common words
    max_features=1000       # only keep top 1000 important words
)

tfidf = vectorizer.fit_transform(documents)

# --- Step 2: Fit NMF model ---
num_topics = 10   # number of topics
nmf_model = NMF(n_components=num_topics, random_state=42, max_iter=500)
nmf_model.fit(tfidf)

# --- Step 3: Display the topics ---
# Get the words corresponding to features
feature_names = vectorizer.get_feature_names_out()

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}:")
        print(" | ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
        print()

# Show top 10 words per topic
display_topics(nmf_model, feature_names, no_top_words=15)

import numpy as np
# Assign the topic with the highest probability
topic_distributions = nmf_model.transform(tfidf)

assigned_topics = np.argmax(topic_distributions, axis=1)

# Add the assigned topic back to your original dataframe
philly_restaurant_reviews['topic'] = assigned_topics

# pie chart of the topic column
philly_restaurant_reviews['topic'].value_counts().plot(kind='pie')

# Example reviews (replace with your real reviews later!)
documents = philly_balanced_clean['clean_review'].tolist()

# --- Step 1: Convert the reviews to TF-IDF vectors ---
vectorizer = TfidfVectorizer(
    stop_words='english',   # remove common words
    max_features=1000       # only keep top 1000 important words
)

tfidf = vectorizer.fit_transform(documents)

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}:")
        print(" | ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
        print()

nmf_models = []

for i in [5, 7, 10, 20]:
  # --- Step 2: Fit NMF model ---
  num_topics = i   # number of topics
  nmf_model = NMF(n_components=num_topics, random_state=42, max_iter=500)
  nmf_model.fit(tfidf)
  nmf_models.append((num_topics, nmf_model))

  # --- Step 3: Display the topics ---
  # Get the words corresponding to features
  feature_names = vectorizer.get_feature_names_out()

  # Show top 10 words per topic
  print("Num Topics:", num_topics)
  display_topics(nmf_model, feature_names, no_top_words=10)

import joblib
# save vectorizer
joblib.dump(vectorizer, f"/yelp-dataset/nmf_model/tfidf_vectorizer.pkl")

# Save the models
for num_topics, nmf_model in nmf_models:
  joblib.dump(nmf_model, f"/yelp-dataset/nmf_model/nmf_model_topics_{num_topics}.pkl")

import numpy as np

for num_topics, nmf_model in nmf_models:
  # Assign the topic with the highest probability
  topic_distributions = nmf_model.transform(tfidf)

  assigned_topics = pd.DataFrame(np.argmax(topic_distributions, axis=1), columns=['topic'])

  assigned_topics.to_json(f'/content/drive/MyDrive/yelp-dataset/nmf_model/philly_nmf_topics_{num_topics}.json', orient='records', lines=True)

# pie chart of the topic column
philly_balanced_clean['topic'].value_counts().plot(kind='pie');

# === 1. Load your reviews ===
# Replace this with your own list of reviews
reviews = documents

# === 2. Get sentence embeddings ===
model = SentenceTransformer('all-MiniLM-L6-v2')  # small and fast
embeddings = model.encode(reviews)

# === 3. (Optional but helpful) Dimensionality reduction with UMAP ===
umap_model = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
umap_embeddings = umap_model.fit_transform(embeddings)

# === 4. Clustering with HDBSCAN ===
clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean', cluster_selection_method='eom')
cluster_labels = clusterer.fit_predict(umap_embeddings)

# === 5. Summarize clusters into topics ===
# Group reviews by their cluster labels
df = pd.DataFrame({'review': reviews, 'cluster': cluster_labels})

# Vectorize all the reviews
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['review'])

# For each cluster, find top words
def get_top_words_per_cluster(X, labels, vectorizer, top_n=5):
    results = {}
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label == -1:
            continue  # -1 means noise (unclustered points)
        idx = np.where(labels == label)[0]
        cluster_X = X[idx]
        mean_tfidf = np.asarray(cluster_X.mean(axis=0)).flatten()
        top_indices = mean_tfidf.argsort()[::-1][:top_n]
        top_words = [vectorizer.get_feature_names_out()[i] for i in top_indices]
        results[label] = top_words
    return results

topics = get_top_words_per_cluster(X, cluster_labels, vectorizer)

# === 6. Print out the topics ===
for cluster, words in topics.items():
    print(f"Cluster {cluster}: {', '.join(words)}")

# Also, show the clustering results
print("\nClustered Reviews:")
print(df)

from sentence_transformers import SentenceTransformer, util

# Topics you care about
topics = ["food", "service", "ambience", "price", "location", "staff", "drinks", "cleanliness"]

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode topic keywords
topic_embeddings = model.encode(topics)

# Encode reviews
reviews = philly_restaurant_reviews.sample(n=10, random_state=42)['review'].tolist()
review_embeddings = model.encode(reviews)

# For each review, find matching topics
review_topics = []

for review_emb in review_embeddings:
    similarities = util.cos_sim(review_emb, topic_embeddings)[0]
    matches = [topics[i] for i, score in enumerate(similarities) if score >= 0.15]  # Threshold can be tuned
    review_topics.append(matches)

# Print results
for review, topics_found in zip(reviews, review_topics):
    print(f"\nReview: {review}\nTopics: {topics_found}")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Topics you care about
topics = ["food", "service", "ambience", "price", "location", "staff", "drinks", "cleanliness"]

# Reviews
reviews = philly_restaurant_reviews.sample(n=10, random_state=42)['review'].tolist()

# Combine topics + reviews to build vocabulary
corpus = topics + reviews

# Fit TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
tfidf_matrix = vectorizer.fit_transform(corpus)

# Split back out
topic_vectors = tfidf_matrix[:len(topics)]
review_vectors = tfidf_matrix[len(topics):]

# For each review, find matching topics
review_topics = []

for review_vec in review_vectors:
    sims = cosine_similarity(review_vec, topic_vectors)[0]
    matches = [topics[i] for i, score in enumerate(sims) if score > 0.0]  # threshold, can tune
    review_topics.append(matches)

# Print results
for review, topics_found in zip(reviews, review_topics):
    print(f"\nReview: {review}\nTopics: {topics_found}")

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Topics you care about
topics = ["food", "service", "ambience", "price", "location", "staff", "drinks", "cleanliness"]

# Reviews
reviews = philly_restaurant_reviews.sample(n=10, random_state=42)['review'].tolist()

# Combine topics + reviews to build vocabulary
corpus = topics + reviews

# Fit BoW (CountVectorizer)
vectorizer = CountVectorizer(stop_words='english')
bow_matrix = vectorizer.fit_transform(corpus)

# Split back out
topic_vectors = bow_matrix[:len(topics)]
review_vectors = bow_matrix[len(topics):]

# For each review, find matching topics
review_topics = []

for review_vec in review_vectors:
    sims = cosine_similarity(review_vec, topic_vectors)[0]
    matches = [topics[i] for i, score in enumerate(sims) if score > 0.2]  # any overlap
    review_topics.append(matches)

# Print results
for review, topics_found in zip(reviews, review_topics):
    print(f"\nReview: {review}\nTopics: {topics_found}")

from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd

# Reviews
reviews = philly_restaurant_reviews.sample(n=100_000, random_state=42)['review'].tolist()

# === 1. Create Bag of Words ===
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(reviews)

# === 2. Fit LDA model ===
n_topics = 10  # You can change this number depending on your data
lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda_model.fit(X)

# === 3. Show topics (top words per topic) ===
words = vectorizer.get_feature_names_out()

def print_topics(model, feature_names, n_top_words=5):
    for topic_idx, topic in enumerate(model.components_):
        top_features = topic.argsort()[:-n_top_words - 1:-1]
        topic_words = [feature_names[i] for i in top_features]
        print(f"Topic {topic_idx}: {', '.join(topic_words)}")

print_topics(lda_model, words)

# === 4. Assign topics to reviews ===
# Get topic distribution for each review
topic_distributions = lda_model.transform(X)

# Create DataFrame for easier viewing
df = pd.DataFrame(topic_distributions, columns=[f"Topic {i}" for i in range(n_topics)])
df['Review'] = reviews

# Show the topic with highest probability for each review
df['Assigned Topic'] = df[[f"Topic {i}" for i in range(n_topics)]].idxmax(axis=1)

# print("\nReview Assignments:")
# print(df[['Review', 'Assigned Topic']])

print_topics(lda_model, words, n_top_words=10)

df['Assigned Topic'].groupby(df['Assigned Topic']).count()

# get rows with topic 7
df[df['Assigned Topic'] == 'Topic 7']

df[df['Assigned Topic'] == 'Topic 7'].iloc[0]['Review']




# Your reviews
reviews = philly_restaurant_reviews.sample(n=1_000, random_state=42)['review'].tolist()

# Create BERTopic model
topic_model = BERTopic(language="english", verbose=True)

# Fit model
topics, probs = topic_model.fit_transform(reviews)

# Show found topics
topic_info = topic_model.get_topic_info()
print("\n=== Topics ===")
print(topic_info)

# vialize topics
topic_model.visualize_topics().show()

from scipy.stats import ttest_ind

topic1 = df[df['topic'] == 1]['stars']
others = df[df['topic'] != 1]['stars']

# One-sided t-test: is Topic 1 mean less than others?
t_stat, p_val = ttest_ind(topic1, others, equal_var=False)

print(f"T-statistic: {t_stat:.3f}, P-value (two-sided): {p_val:.4f}")
print(f"P-value (one-sided, lower): {p_val/2:.4f}" if t_stat < 0 else "No evidence Topic 1 is lower.")