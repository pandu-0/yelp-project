# Yelp Project

This repository contains the code and resources for our **Restaurant Review Project**. The aim of this project is to analyze customer reviews of restaurants and measure quality in a quantifiable manner. Using natural language processing (e.g., sentiment analysis), we examine reviews to uncover satisfaction patterns and identify key concerns. We use average review scores, sentiment polarity, and topic modeling across Yelp review datasets to provide insights for both customers and businesses.

**Dataset:** [Yelp Dataset](https://business.yelp.com/data/resources/open-dataset/#:~:text=Documentation%20is%20included.-,Download%20JSON,-This%20download%20contains)

> [!NOTE]  
We suggest directly downloading the dataset from the link above and placing in your working directory instead of downloading
it using the `opendatasets` library in our scripts. This will eliminate the need for the Kaggle API key.

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
    pip install -r code/requirements.txt
    ```



## ML Pipeline
**Step 1:** The pipline begins with the preprocessing script where we take the original dataset and filter the dataset down to only include Phildelphia reviews. 

**Step 2:** Then we move onto the EDA script where we used the dataset created in the previous step to conduct exploratory data analysis to learn more about the data and what we're working with. 

**Step 3:** Once we finished our EDA, we did some more preprocessing, topic modeling, and testing in the topic modeling and BERTopic testing scripts.

**Step 4:** After the topic modeling, we did some sentiment analysis in the sentiment analysis script. 

**Step 5:** Finally to wrap things up, we did some conlusionary analysis based on the sentiment analysis and topic modeling results from the previous steps. 

**Step 6:** After wrapping our analysis up, we proceeded to build and deploy our website, which consists of 4 pages: Home, EDA, Analysis, and Major Findings.

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
The above directory tree depicts the structure of the `appengine` directory. The `appengine` directory hosts all the code for the website to be used by Google Cloud App Engine API.

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
└── README.md                          # Project documentation
```
The above directory tree depicts the structure of the assets folder, which contains the datasets and sample datasets used in making the website. 

### Preprocessing/Analysis Code
```
├── code/                                 # Python scripts for data analysis and modeling
│   ├── analyze_models.py                     # Script for comparing and analyzing different models
│   ├── bertopic_modeling.py                  # Script for BERTopic topic modeling
│   ├── acquire_dataset.py                    # Script for acquiring and preprocessing the Yelp dataset
│   ├── eda.py                                # Script for exploratory data analysis (EDA)
│   ├── nmf_modeling.py                       # Script for NMF topic modeling
│   ├── correlation.py  # Script analyzing correlations between open status, adjectives, and ratings
│   ├── sentiment_analysis.py                 # Script for sentiment analysis on Yelp reviews
```

Scripts used for data acquisition, exploratory analysis, topic modeling (BERTopic & NMF), sentiment analysis, and correlation studies in the Yelp review pipeline can be found in the `code` directory.

> [!IMPORTANT]  
Although all scripts can be run on a local machine, we highly suggest executing them on Google Colab since it comes with most packages pre-installed except for `bertopic`. You can also avoid package dependency problems in this way.

## Website Link
https://cs163-project-452620.wn.r.appspot.com/

## Contact
For questions please contact:

Pandu Rupanagudi : [Linkedin](https://www.linkedin.com/in/manmohanbabu/)  
Singaram Subramanyan : [LinkedIn](https://www.linkedin.com/in/singaram-s01/)

