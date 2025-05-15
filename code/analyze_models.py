# Import libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import numpy as np
import seaborn as sns
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

philly_balanced_clean = pd.read_json('/yelp-dataset/philly_balanced_with_sentiment_clean.json', lines=True)

documents = philly_balanced_clean['clean_review'].tolist()

# --- Step 1: Convert the reviews to TF-IDF vectors ---
vectorizer = TfidfVectorizer(
    stop_words='english',   # remove common words
    max_features=1000       # only keep top 1000 important words
)

tfidf = vectorizer.fit_transform(documents)

# --- Step 2: Fit NMF model ---
num_topics = 7   # number of topics
nmf_model = NMF(n_components=num_topics, random_state=42, max_iter=500)
nmf_model.fit(tfidf)

topic_distributions = nmf_model.transform(tfidf)

assigned_topics = pd.DataFrame(np.argmax(topic_distributions, axis=1), columns=['topic'])
philly_balanced_clean['topic'] = assigned_topics

# set sentiment label as column using argmax
philly_balanced_clean['sentiment_label'] = philly_balanced_clean[['pos_prob', 'neu_prob', 'neg_prob']].idxmax(axis=1)

philly_balanced_clean

philly_balanced_clean.groupby(['topic', 'is_open'])['stars'].mean()



# Step 1: Group and reset index
grouped = philly_balanced_clean.groupby(['topic', 'is_open'])['stars'].mean().reset_index()

# Step 2: Create the barplot
plt.figure(figsize=(10, 6))
sns.barplot(data=grouped, x='topic', y='stars', hue='is_open', palette='Set2')

# Step 3: Beautify
plt.title('Average Star Rating by Topic and Open Status')
plt.xlabel('Topic')
plt.ylabel('Mean Stars')
plt.legend(title='Is Open')  # Just set the title, no custom labels
plt.ylim(2.5, 4.6)  # Adjust based on your data
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('/content/assets/mean_rating_over_topic_7_open_status_plot.svg', format='svg')
plt.show()

philly_balanced_clean.groupby(['is_open'])['topic'].value_counts().sort_index()

philly_balanced_clean.groupby(['sentiment_label'])['topic'].value_counts().sort_index()

sns.countplot(data=philly_balanced_clean, x='topic', hue='is_open')

philly_balanced_clean['date'] = pd.to_datetime(philly_balanced_clean['date'])

# Create the 'year' and 'month' columns
philly_balanced_clean['year'] = philly_balanced_clean['date'].dt.year
philly_balanced_clean['month'] = philly_balanced_clean['date'].dt.month

# Check the result
philly_balanced_clean.head()

philly_balanced_clean.groupby(['year', 'topic'])['stars'].mean()


# font config
fconfig = {'font.size': 14, 'font.family': 'serif'}
matplotlib.rcParams.update(fconfig)

# Step 1: Group by year, topic, and is_open
grouped = philly_balanced_clean.groupby(['year', 'topic', 'is_open'])['stars'].mean().reset_index()

# Step 2: Plot with one line per is_open value in each topic subplot
g = sns.relplot(
    data=grouped,
    x='year',
    y='stars',
    col='topic',
    hue='is_open',          # Lines for open (1) and closed (0)
    kind='line',
    col_wrap=4,
    height=3.5,
    aspect=1.2,
    marker='o',
    palette='tab10'
)

# Step 3: Clean x-axis: show ticks every 4 years
years = sorted(grouped['year'].unique())
tick_years = [year for year in years if year % 4 == 0]

for ax in g.axes.flat:
    ax.set_xticks(tick_years)
    ax.set_xticklabels(tick_years, rotation=45)

g.set_titles("Topic {col_name}")
g.set_axis_labels("Year", "Mean Stars")
g._legend.set_title("Is Open")  # Rename legend title
g.tight_layout()
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Mean Star Rating Over Time by Topic and Open Status')
fig_path = "/content/assets/mean_rating_over_year_topic_7_plot.svg"
plt.savefig(fig_path, format="svg")
plt.show()


# font config
fconfig = {'font.size': 14, 'font.family': 'serif'}
matplotlib.rcParams.update(fconfig)

# Step 1: Group by year, topic, and is_open
grouped = philly_balanced_clean.groupby(['year', 'topic', 'is_open'])['stars'].mean().reset_index()

# Step 2: Plot with one line per is_open value in each topic subplot
g = sns.relplot(
    data=grouped,
    x='year',
    y='stars',
    col='topic',
    hue='is_open',          # Lines for open (1) and closed (0)
    kind='line',
    col_wrap=4,
    height=3.5,
    aspect=1.2,
    marker='o',
    palette='tab10'
)

# Step 3: Clean x-axis: show ticks every 4 years
years = sorted(grouped['year'].unique())
tick_years = [year for year in years if year % 4 == 0]

for ax in g.axes.flat:
    ax.set_xticks(tick_years)
    ax.set_xticklabels(tick_years, rotation=45)

g.set_titles("Topic {col_name}")
g.set_axis_labels("Year", "Mean Stars")
g._legend.set_title("Is Open")  # Rename legend title
g.tight_layout()
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Mean Star Rating Over Time by Topic and Open Status')
fig_path = "/content/assets/mean_rating_over_topic_7_open_status_plot.svg"
plt.savefig(fig_path, format="svg")
plt.show()

# find the most common topic for each business and set it as a column
philly_balanced_clean['most_common_topic'] = philly_balanced_clean.groupby('business_id')['topic'].transform(lambda x: x.value_counts().idxmax())

philly_restaurants = pd.read_json("/content/drive/MyDrive/yelp-dataset/philly_restaurants.json/part-00000-f4870cc6-8bc2-4742-8cb3-d71bb3c83899-c000.json",
                                  lines = True)

philly_restaurants[['business_id', 'latitude', 'longitude', 'stars']].merge(philly_balanced_clean[['business_id', 'most_common_topic']], on='business_id', how='left')

# give me unique business and their corresponding latitude longitude and most_comoon topic
philly_business_topics = philly_restaurants[['business_id', 'latitude', 'longitude', 'stars']].merge(philly_balanced_clean[['business_id', 'most_common_topic']], on='business_id', how='left').drop_duplicates()

# Drop rows where most_common_topic is missing
df = philly_business_topics.dropna(subset=['most_common_topic'])

# Optional: convert topic to string if you want categorical coloring
df['most_common_topic'] = df['most_common_topic'].astype(str)

# Create the map
fig = px.scatter_mapbox(
    df,
    lat='latitude',
    lon='longitude',
    color='most_common_topic',
    hover_name='business_id',
    hover_data=['stars'],
    zoom=10,
    height=700,
    title="Philly Businesses Colored by Most Common Topic"
)

# Set the mapbox style and token (you can use "open-street-map" without a token)
fig.update_layout(mapbox_style="open-street-map")

# Show the plot
fig.show()

