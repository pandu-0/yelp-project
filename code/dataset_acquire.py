from pyspark.sql import SparkSession
import opendatasets as od

# Dataset can be downloaded with this:
od.download("https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset", progressbar=True)

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

sc = spark.sparkContext

business_df = spark.read.json("/content/drive/MyDrive/yelp-dataset/yelp_academic_dataset_business.json")
business_df.createOrReplaceTempView("business")

philly_restaurants = spark.sql("""
    SELECT business_id, categories, is_open, latitude, longitude, stars, review_count, name
    FROM business
    WHERE state = 'PA'
    AND city = 'Philadelphia'
    AND categories LIKE '%Restaurants%'
""")

philly_restaurants.createOrReplaceTempView("philly_restaurants")

# already done, no need to run it again
philly_restaurants.coalesce(1).write.json("/yelp-dataset/philly_restaurants.json", mode="overwrite")

business_reviews = spark.read.json("/yelp-dataset/yelp_academic_dataset_review.json")
business_reviews.createOrReplaceTempView("business_reviews")

philly_restaurants_reviews = spark.sql("""
    SELECT philly_restaurants.business_id, philly_restaurants.categories, philly_restaurants.city, philly_restaurants.state, philly_restaurants.is_open, business_reviews.cool, business_reviews.funny, business_reviews.date, business_reviews.stars,
    business_reviews.text AS review, business_reviews.useful, business_reviews.user_id
    FROM business_reviews
    RIGHT JOIN philly_restaurants ON philly_restaurants.business_id = business_reviews.business_id
""")

philly_restaurants_reviews.createOrReplaceTempView("philly_restaurants_reviews")


philly_restaurants_reviews.coalesce(1).write.json("/content/drive/MyDrive/yelp-dataset/philly_restaurants_reviews.json", mode="overwrite")
