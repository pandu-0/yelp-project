# Import libraries
import pandas as pd
from tqdm.notebook import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from bertopic import BERTopic

philly_balanced_clean = pd.read_json('/yelp-dataset/philly_balanced_with_sentiment_clean.json', lines=True)

# Sample reviews
reviews_sampled = philly_balanced_clean

# Convert sampled reviews to list
reviews = reviews_sampled['clean_review'].tolist()

# Create BERTopic model
topic_model = BERTopic(language="english", verbose=True)

# Fit model
topics, probs = topic_model.fit_transform(reviews)

# Add topics as a new column in the sampled DataFrame
reviews_sampled['topic'] = topics

# Show found topics
topic_info = topic_model.get_topic_info()
print("\n=== Topics ===")
print(topic_info)

# Optionally show a few rows
print(reviews_sampled[['clean_review', 'topic']].head())

# Method 2 - pytorch
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
topic_model.save("/content/drive/MyDrive/yelp-dataset/bertopic_models/philly_complete",
                 serialization="pytorch", save_ctfidf=True, save_embedding_model=embedding_model)

# vialize topics
topic_model.visualize_topics().show()

topic_model_load = BERTopic.load("/content/drive/MyDrive/yelp-dataset/bertopic_models/philly_complete")

topics_fig = topic_model_load.visualize_topics()
topics_fig.write_html("/content/drive/MyDrive/yelp-dataset/analysis/topics_fig.html")

topics_fig.show()

heirarchy_tree_fig = topic_model_load.visualize_hierarchy()
heirarchy_tree_fig.write_html("/content/drive/MyDrive/yelp-dataset/analysis/heirarchy_tree_fig.html")

heirarchy_tree_fig.show()

topic_word_scores_fig = topic_model_load.visualize_barchart()
topic_word_scores_fig.write_html("/content/drive/MyDrive/yelp-dataset/analysis/topic_word_scores_fig.html")

topic_word_scores_fig.show()