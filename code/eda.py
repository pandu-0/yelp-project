from pyspark.sql import SparkSession
import plotly.express as px

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

sc = spark.sparkContext

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import seaborn as sns

# REPLACE FILE PATH WITH YOUR OWN
philly_restaurants = pd.read_json("/yelp-dataset/philly_restaurants.json/part-00000-f4870cc6-8bc2-4742-8cb3-d71bb3c83899-c000.json", lines=True)

# REPLACE FILE PATH WITH YOUR OWN
philly_restaurant_reviews = pd.read_json('/yelp-dataset/philly_restaurants_reviews_with_sentiment_complete.json', lines=True)

# REPLACE FILE PATH WITH YOUR OWN
philly_clean = pd.read_json("/content/drive/MyDrive/yelp-dataset/philly_balanced_with_sentiment_clean.json", lines=True)

# Make sure the 'date' column is a datetime type
philly_clean['date'] = pd.to_datetime(philly_clean['date'])

# Create the 'year' and 'month' columns
philly_clean['year'] = philly_clean['date'].dt.year
philly_clean['month'] = philly_clean['date'].dt.month

# Make sure the 'date' column is a datetime type
philly_restaurant_reviews['date'] = pd.to_datetime(philly_restaurant_reviews['date'])

# Create the 'year' and 'month' columns
philly_restaurant_reviews['year'] = philly_restaurant_reviews['date'].dt.year
philly_restaurant_reviews['month'] = philly_restaurant_reviews['date'].dt.month


philly_restaurant_reviews['business_id'].nunique()

stars_describe = philly_restaurant_reviews.stars.describe().round(2).to_frame().reset_index()
stars_describe.columns = ['statistic', 'stars']
print(stars_describe)

stars_count = philly_restaurant_reviews['stars'].value_counts().to_frame().reset_index()
stars_count.to_json("/content/drive/MyDrive/yelp-dataset/eda/stars_count.json", orient='records', lines=True)
print(stars_count)

sns.barplot(x='stars', y='count', data=stars_count)

is_open_count = philly_restaurant_reviews['is_open'].value_counts().to_frame().reset_index()
is_open_count

is_open_count.to_json("/content/drive/MyDrive/yelp-dataset/eda/is_open_count.json", orient='records', lines=True)

sns.barplot(x='is_open', y='count', data=is_open_count);

# average star by business
business_stars = philly_restaurant_reviews.groupby('business_id')['stars'].mean()
business_stars.describe()

business_stars.to_frame().reset_index().to_json("/content/drive/MyDrive/yelp-dataset/eda/restaurant_mean_stars.json", orient='records', lines=True)

sns.violinplot(business_stars, orient='h');

# number of reviews a restaurant has
temp = philly_restaurant_reviews.groupby('business_id')[['review']].count()
temp.describe()



temp_2 = philly_restaurant_reviews.groupby('business_id')[['funny']].sum()
temp_2.head()

temp_complete = pd.concat([temp, temp_2], axis=1)
temp_complete.head()

# remove outliers using IQR for temp_complete in review column
Q1 = temp_complete.review.quantile(0.25)
Q3 = temp_complete.review.quantile(0.75)
IQR = Q3 - Q1

restaurant_review_count = temp_complete[~((temp_complete.review < (Q1 - 1.5 * IQR)) |(temp_complete.review > (Q3 + 1.5 * IQR)))]
restaurant_review_count.describe()

restaurant_review_count

restaurant_review_count.to_json("/content/drive/MyDrive/yelp-dataset/eda/restaurant_review_count.json", orient='records', lines=True)

sns.violinplot(temp_complete['review'], orient='h');

philly_restaurant_reviews[['year', 'is_open']].to_json("/content/drive/MyDrive/yelp-dataset/eda/year_and_is_open.json", orient='records', lines=True)

plt.figure(figsize=(12, 6))  # Make the plot wider
sns.countplot(philly_restaurant_reviews, x='year', hue='is_open')
plt.xticks(rotation=45)      # Rotate x-axis labels 45 degrees
plt.tight_layout()           # Adjust layout to prevent label cutoff
plt.show()

philly_restaurant_reviews.useful.describe()

# remove outliers in userful using IQR
Q1 = philly_restaurant_reviews.useful.quantile(0.25)
Q3 = philly_restaurant_reviews.useful.quantile(0.75)
IQR = Q3 - Q1
useful_describe = philly_restaurant_reviews[~((philly_restaurant_reviews.useful < (Q1 - 1.5 * IQR)) |(philly_restaurant_reviews.useful > (Q3 + 1.5 * IQR)))]


philly_restaurant_reviews.to_json("/content/drive/MyDrive/yelp-dataset/philly_restaurants_reviews_with_sentiment_complete.json", orient='records', lines=True)



# Convert is_open to string so Plotly treats it as categorical
philly_restaurants['is_open_str'] = philly_restaurants['is_open'].map({1: "Open", 0: "Closed"})

fig = px.scatter_mapbox(
    philly_restaurants,
    lat="latitude",
    lon="longitude",
    color="is_open_str",
    hover_name="business_id",
    hover_data={"categories": True},
    zoom=11,
    center={"lat": 39.9526, "lon": -75.1652},
    height=600,
    title="📍 Philadelphia Restaurants: Open vs Closed"
)

fig.update_layout(
    mapbox_style="open-street-map",
    legend_title_text="Status",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.01,
        xanchor="right",
        x=0.99
    ),
    margin=dict(l=10, r=10, t=50, b=10),
    title_x=0.5,
    title_font=dict(size=20, color='black')
)

fig.show()

