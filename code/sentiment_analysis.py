import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib

philly_restaurant_reviews = pd.read_json("/content/drive/MyDrive/yelp-dataset/philly_restaurants_reviews.json", lines=True)

tqdm.pandas()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# RoBERTa-Based Sentiment Analysis
sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name).to(device)

def get_sentiment_probs(text):
    try:
        inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = sentiment_model(**inputs)

        logits = outputs.logits[0]
        probs = F.softmax(logits, dim=0)
        neg_prob = probs[0].item()
        neu_prob = probs[1].item()
        pos_prob = probs[2].item()
    except Exception as e:
        print(f"Broke on review: {text[:100]}... | Error: {str(e)}")
        neg_prob, neu_prob, pos_prob = 0.0, 0.0, 0.0

    return neg_prob, neu_prob, pos_prob

# Apply the function with a progress bar
philly_restaurant_reviews[['neg_prob', 'neu_prob', 'pos_prob']] = philly_restaurant_reviews['review'].progress_apply(
    lambda x: pd.Series(get_sentiment_probs(x))
)

philly_restaurant_reviews.to_json('/yelp-dataset/philly_restaurants_reviews_with_sentiment_complete.json', 
                                  orient='records', lines=True
)


## Balanced dataset sentiment analysis

philly_balanced_clean = pd.read_json('/content/drive/MyDrive/yelp-dataset/philly_balanced_with_sentiment_clean.json', lines=True)
# Make sure the 'date' column is a datetime type
philly_balanced_clean['date'] = pd.to_datetime(philly_balanced_clean['date'])

# Create the 'year' and 'month' columns
philly_balanced_clean['year'] = philly_balanced_clean['date'].dt.year
philly_balanced_clean['month'] = philly_balanced_clean['date'].dt.month

philly_balanced_clean[['business_id','is_open', 'year', 'month','stars','neg_prob', 'neu_prob', 'pos_prob']].to_json(
    '/content/drive/MyDrive/yelp-dataset/philly_balanced_with_sentiment_probs_and_no_review.json', 
    orient='records', lines=True
)


# font config
fconfig = {'font.size': 14, 'font.family': 'serif'}
matplotlib.rcParams.update(fconfig)

fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=philly_balanced_clean, x='stars', y='pos_prob', ax=axs[0], hue='stars', palette="Paired")
sns.barplot(data=philly_balanced_clean, x='stars', y='neu_prob', ax=axs[1], hue='stars', palette="Paired")
sns.barplot(data=philly_balanced_clean, x='stars', y='neg_prob', ax=axs[2], hue='stars', palette="Paired")

for i in range(3):
  axs[i].set_yticks([0, 0.5, 1.0])
  axs[i].get_legend().remove()

axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
fig_path = "/content/assets/sentiment_plot.svg"
plt.savefig(fig_path, format="svg")
plt.show()
plt.close()