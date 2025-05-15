# Import libraries
from sklearn.decomposition import NMF
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from sklearn.decomposition import LatentDirichletAllocation
from scipy.stats import ttest_ind

# REPLACE FILE PATH WITH YOUR OWN
philly_balanced_clean = pd.read_json('/yelp-dataset/philly_balanced_with_sentiment_clean.json', lines=True)

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


# save vectorizer
joblib.dump(vectorizer, f"/yelp-dataset/nmf_model/tfidf_vectorizer.pkl")

# Save the models
for num_topics, nmf_model in nmf_models:
  joblib.dump(nmf_model, f"/yelp-dataset/nmf_model/nmf_model_topics_{num_topics}.pkl")

for num_topics, nmf_model in nmf_models:
  # Assign the topic with the highest probability
  topic_distributions = nmf_model.transform(tfidf)

  assigned_topics = pd.DataFrame(np.argmax(topic_distributions, axis=1), columns=['topic'])

  assigned_topics.to_json(f'/content/drive/MyDrive/yelp-dataset/nmf_model/philly_nmf_topics_{num_topics}.json', orient='records', lines=True)

# pie chart of the topic column
philly_balanced_clean['topic'].value_counts().plot(kind='pie');


# === LDA TESTING ===
# Reviews
reviews = philly_balanced_clean['clean_review'].tolist()

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


topic1 = df[df['topic'] == 1]['stars']
others = df[df['topic'] != 1]['stars']

# One-sided t-test: is Topic 1 mean less than others?
t_stat, p_val = ttest_ind(topic1, others, equal_var=False)

print(f"T-statistic: {t_stat:.3f}, P-value (two-sided): {p_val:.4f}")
print(f"P-value (one-sided, lower): {p_val/2:.4f}" if t_stat < 0 else "No evidence Topic 1 is lower.")

