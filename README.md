# Yelp Project

This repository contains the code and resources for our **Restaurant Review Project**. The aim of this project is to analyze customer reviews of restaurants and measure quality in a quantifiable manner. Using natural language processing (e.g., sentiment analysis), we examine reviews to uncover satisfaction patterns and identify key concerns. We use average review scores, sentiment polarity, and topic modeling across Yelp review datasets to provide insights for both customers and businesses.

**Dataset:** https://business.yelp.com/data/resources/open-dataset/

## Setup
1. Clone the repository:
    ```bash
    git clone https://github.com/pandu-0/yelp-project.git
    ```
2. Navigate to the project directory:
    ```bash
    cd yelp-project
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
### Required Packages

The following Python packages are required for this project:

- `dash`
- `plotly`
- `pyspark`
- `pandas`
- `Flask`
- `gunicorn`
- `google-cloud-storage`
- `joblib`
- `scikit-learn`

These packages can be installed using the `requirements.txt` file provided.

## ML Pipeline
**Step 1:** The pipline begins with the preprocessing script where we take the original dataset and filter the dataset down to only include Phildelphia reviews. 
**Step 2:** Then we move onto the EDA script where we used the dataset created in the previous step to conduct exploratory data analysis to learn more about the data and what we're working with. 
**Step 3:** Once we finished our EDA, we did some more preprocessing, topic modeling, and testing in the topic modeling and BerTopic testing scripts.
**Step 4:** After the topic modeling, we did some sentiment analysis in the sentiment analysis script. 
**Step 5:** Finally to wrap things up, we did some conlusionary analysis based on the sentiment analysis and topic modeling results from the previous steps. 

## Directory Information

### AppEngine
```
├── appengine/                  # Google App Engine configuration and deployment files
│   ├── app.yaml                # App Engine configuration file
│   ├── main.py                 # Entry point for the web application
│   ├── requirements.txt        # Python dependencies
│   ├── utils.py                # Utility functions
│   └── pages/                  # Website page folder
│       ├── analysis.py         # Analysis page
│       ├── eda.py              # EDA page
│       ├── major_findings.py   # Major findings page
│       └── home.py             # Home page

```
The above directory tree depicts the structure of the appengine folder. The appengine folder hosts all the code for the website. The main.py file should be run to see the website in debug mode. 

### Assets
```

├── assets/                            # Files for website datasets
│   ├── Adjective_correlation_rating/  # Adjective correlation rating data
│   ├── business_head.json             # Sample business data
│   ├── checkin_head.json              # Sample check-in data
│   ├── mean_rating_over_topic_7_open_status_plot.svg  # Plot for mean rating over topic 7 (open status)
│   ├── mean_rating_over_year_topic_7_plot.svg  # Plot for mean rating over year (topic 7)
│   ├── review_head.json               # Sample review data
│   ├── sentiment_plot.svg             # Sentiment analysis plot
│   ├── tips_head.json                 # Sample tips data
│   └── user_head.json                 # Sample user data
└── ReadMe.md                          # Project documentation
```
The above directory tree depicts the structure of the assets folder, which contains the datasets and sample datasets used in making the website. 

### Preprocessing/Analysis Code
```

Preprocessing/Analysis Code directory tree goes here
```

## Website Link
https://cs163-project-452620.wn.r.appspot.com/

## Contact
For questions please contact  .