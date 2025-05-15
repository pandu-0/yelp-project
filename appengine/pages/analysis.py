import dash
from dash import html, dcc, dash_table, Input, Output, callback
import pandas as pd
import plotly.express as px
from utils import get_json_from_gcs, load_pickle_from_gcs, GCLOUD_BUCKET

dash.register_page(__name__, path='/analysis-methods', name='Analysis')

adjective_corr_df = pd.read_csv("https://raw.githubusercontent.com/pandu-0/yelp-project/refs/heads/main/assets/Adjective_correlation_rating")

adjective_corr_df["Correlation"] = adjective_corr_df["Correlation"].round(2)
adjective_corr_df["average_rating"] = adjective_corr_df["average_rating"].round(2)

top_5 = adjective_corr_df.sort_values(by="Correlation", ascending=False).head(5)
bottom_5 = adjective_corr_df.sort_values(by="Correlation", ascending=False).tail(5)

adj_corr_table_rows = [
    html.Tr([html.Td(row["Adjective"]), html.Td(row["Correlation"]), html.Td(row["average_rating"])])
    for i, row in pd.concat([top_5, bottom_5]).iterrows()
]

# load vectorizer
vectorizer = load_pickle_from_gcs(GCLOUD_BUCKET, "models/nmf_model/tfidf_vectorizer.pkl")

topic_keywords = dict()

# Load the saved model
for num_topics in [5, 7, 10, 20]:
    nmf_model = load_pickle_from_gcs(GCLOUD_BUCKET, f"models/nmf_model/nmf_model_topics_{num_topics}.pkl")

    # Extract feature names (vocabulary words)
    feature_names = vectorizer.get_feature_names_out()

    # Show top N words per topic
    no_top_words = 10
    nmf_topics = [
        [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        for topic in nmf_model.components_
    ]

    topic_keywords[num_topics] = [" | ".join(words) for words in nmf_topics]

philly_balanced_clean = get_json_from_gcs(GCLOUD_BUCKET, "analysis/philly_balanced_with_sentiment_probs_and_no_review.json")

@callback(
    Output('nmf-topics', 'data'),
    Input('nmf-topic-radio', 'value')
)
def update_nmf_table(topic_count):
    return [{"Topic": f"Topic {i}", "Words": kw} for i, kw in enumerate(topic_keywords[topic_count])]


@callback(
    Output("nmf-topics-pie", "figure"),
    Input("nmf-topic-radio", "value")
)
def update_nmf_pie(num_topic):
    philly_nmf_topics = get_json_from_gcs(GCLOUD_BUCKET, f"models/nmf_model/philly_nmf_topics_{num_topic}.json")

    topic_counts = philly_nmf_topics.value_counts().sort_values(ascending=False)  # Sort by count, not index

    fig = px.pie(
        values=topic_counts.values,
        names=[f"Topic {i}" for i in topic_counts.index],  # Descriptive labels
        title=None,
        hole=0.3
    )

    fig.update_layout(
        legend_title_text="Topic Number"
    )
    
    return fig

layout = html.Div([
    html.H1("Analysis Methods", style={
        "fontWeight": 900,
        "marginBottom": "8px",
        "color": "#a94442",
        "fontSize": "2.7rem",
        "fontFamily": "'Poppins', sans-serif",
        "letterSpacing": "1.5px",
        "textShadow": "1px 2px 8px #fbeee6"
    }),

    # Adjective Correlation Section
    html.H3("Adjective Correlation with Ratings", style={
        "color": "#a94442",
        "marginTop": 0,
        "fontWeight": 800,
        "borderLeft": "6px solid #a94442",
        "paddingLeft": "1rem",
        "fontSize": "1.5rem"
    }),
    html.Div([
        html.P(
            "This analysis aimed to determine if there was a correlation between the presence of certain adjectives "
            "and the review rating. "
            "This was accomplished through using the nltk to tokenize each review and filtering for adjectives using "
            "POS (Part of Speech) tagging. "
            "Once filtered down to adjectives, the Pearson correlation coefficient between adjectives and the rating was calculated. "
            "The table below depicts the 5 most and least correlated adjectives."
        )
    ], style={
        "padding": "0 5%",
        "marginBottom": "20px"
    }),
    html.Details([
        html.Summary("Show Code"),
        dcc.Markdown("""
        ```python
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
        ```
        """)
    ], style={"padding": "0 5%", "marginBottom": "30px"}),

    html.Div([
        html.Table([
            html.Thead(
                html.Tr([
                    html.Th("Adjective"),
                    html.Th("Correlation Coefficient"),
                    html.Th("Average Rating")
                ])
            ),
            html.Tbody(adj_corr_table_rows)
        ], style={
            "width": "80%",
            "margin": "0 auto",
            "borderCollapse": "collapse",
            "textAlign": "left"
        }),
        html.P(
            "The average rating column represents the average rating of the reviews with the particular adjective. "
            "As per the above table positively correlated adjectives tend to have a high average rating, and the converse "
            "for negatively correlated adjectives. "
            "Moreover, positively correlated adjectives tend to be adjectives with a positive connotation, such as 'great', "
            "'good', and 'amazing,' while the opposite is true for adjectives with a negative connotation. "
            "This indicates that the presence of a positive connotation adjective increases the chances of receiving a "
            "higher rating, while the presence of a negative connotation adjective increases the chances of "
            "receiving a lower rating. "
        )
    ], style={
        "padding": "0 5%",
        "marginBottom": "40px"
    }),

    # Preprocessing Section
    html.H2("Preprocessing", style={
        "fontWeight": 900,
        "color": "#a94442",
        "fontSize": "2rem",
        "fontFamily": "'Poppins', sans-serif",
        "letterSpacing": "1px",
        "marginTop": "40px",
        "borderLeft": "6px solid #a94442",
        "paddingLeft": "1rem"
    }),
    html.Ul([
        html.Li("The Yelp dataset was mostly clean with minimal null values and duplicates."),
        html.Li("There was a class imbalance between open and closed restaurants."),
        html.Li("To address this, open restaurants were under-sampled to match the number of closed ones."),
        html.Li("Text data was cleaned by removing stop words, punctuation, and applying lemmatization."),
        html.Li("Manual cleaning was chosen over TF-IDF to reduce dataset size."),
        html.Li("We noticed cleaning helped improve the clarity of the topics identified by the various models.")
    ], style={
        "padding": "0 5%",
        "marginBottom": "10px"
    }),
    html.Details([
        html.Summary("Show Preprocessing Code"),
        dcc.Markdown("""
        ```python
        import spacy
        import pandas as pd
        import re

        # balance the dataset
        philly_closed = philly_restaurant_reviews[philly_restaurant_reviews['is_open'] == 0]
        philly_open = philly_restaurant_reviews[philly_restaurant_reviews['is_open'] == 1].sample(len(philly_closed), random_state=42)
        philly_balanced = pd.concat([philly_closed, philly_open], axis=0)

        # Load the small English model
        nlp = spacy.load('en_core_web_sm')

        def advanced_clean_text(text):
            text = str(text).lower()  # Lowercase
            text = re.sub(r'<.*?>', ' ', text)  # Remove HTML tags
            text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and numbers
            text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces

            doc = nlp(text)
            tokens = [
                token.lemma_ for token in doc 
                if not token.is_stop and not token.is_punct and token.is_alpha
            ]
            return ' '.join(tokens)

        # Apply it to your review column
        philly_balanced['clean_review'] = philly_balanced['review'].apply(advanced_clean_text)
        ```
        """)
    ], style={"padding": "0 5%", "marginBottom": "30px"}),

    # Sentiment Analysis Section
    html.H2("Sentiment Analysis", style={
        "fontWeight": 900,
        "color": "#a94442",
        "fontSize": "2rem",
        "fontFamily": "'Poppins', sans-serif",
        "letterSpacing": "1px",
        "marginTop": "40px",
        "borderLeft": "6px solid #a94442",
        "paddingLeft": "1rem",
        "textAlign": "center"
    }),
    html.Div([
        html.H3("Sentiment Probability by Star Rating", style={
            "color": "#a94442",
            "fontWeight": 800,
            "fontSize": "1.3rem",
            "marginTop": "10px"
        }),
        html.Img(
            src="https://raw.githubusercontent.com/pandu-0/yelp-project/refs/heads/main/assets/sentiment_plot.svg",
            style={'width': '100%', 'height': 'auto', "marginBottom": "20px"}
        )
    ], style={"padding": "0 5%"}),
    html.P(
        "The above plot shows the distribution of sentiment probabilities for each star rating. "
        "We can see that the model works because it assigns higher positivity probability to high "
        "ratings and lower positivity probability to low ratings. "
        "This is a good sign that the model is working as intended; however, we can notice that neutral probability of reviews "
        "across all ratings is negligible. This confirms that the dataset has sampling bias, meaning that only customers "
        "who were either really satisfied or really dissatisfied have decided to leave a review for the restaurant they visited",
        style={"padding": "0 5%", "marginBottom": "40px"}
    ),

    html.Details([
        html.Summary("Show Code"),
        dcc.Markdown("""
        ```python
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
        ```
        """)
    ], style={"padding": "0 5%", "marginBottom": "30px"}),

    # Topic Modeling Section
    html.H2("Topic Modeling", style={
        "fontWeight": 900,
        "color": "#a94442",
        "fontSize": "2rem",
        "fontFamily": "'Poppins', sans-serif",
        "letterSpacing": "1px",
        "marginTop": "40px",
        "borderLeft": "6px solid #a94442",
        "paddingLeft": "1rem",
        "textAlign": "center"
    }),
    html.P(
        "Topic modeling is a powerful tool for uncovering hidden themes or patterns in large "
        "collections of text. By automatically grouping words into topics based on how frequently they appear together, "
        "it helps summarize and interpret unstructured data like customer reviews, articles, or social media posts. "
        "This allows analysts to quickly identify what people are talking about, track sentiment around key topics, "
        "and make data-driven decisions without manually reading thousands of documents. In the context of Yelp reviews, "
        "topic modeling reveals common themes customers mention—such as service, food, or cleanliness—and how these relate "
        "to overall satisfaction.",
        style={"padding": "0 5%", "marginBottom": "20px"}
    ),

    # BERTopic Section
    html.H3("BERTopic Analysis", style={
        "color": "#a94442",
        "fontWeight": 800,
        "fontSize": "1.5rem",
        "borderLeft": "6px solid #a94442",
        "paddingLeft": "1rem",
        "marginTop": "30px"
    }),
    html.P(
        "BERTopic is a topic modeling technique that leverages transformer-based language models like BERT "
        "to generate more meaningful and context-aware topics from text. Unlike traditional methods that rely on word "
        "frequency, BERTopic uses dense embeddings to capture semantic relationships between words and documents. "
        "It then applies dimensionality reduction and clustering to group similar documents and extract representative "
        "keywords for each topic. This results in more coherent and interpretable topics, especially useful for analyzing "
        "unstructured text like reviews, feedback, or social media data. We use BERTopic in our project in an attempt to "
        "identify common topics.",
        style={"padding": "0 5%"}
    ),
    html.Div([
        html.Iframe(
            src="https://storage.googleapis.com/cs163-project-452620.appspot.com/analysis/topics_fig.html",
            style={
                "width": "100%",
                "height": "800px",
                "border": "none",
                "overflow": "hidden"
            }
        ),
        html.P(
            "The intertopic distance map visualizes how distinct or similar topics are in a two-dimensional space, "
            "with larger bubbles representing more frequent topics. Closely positioned topics share more semantic similarity, "
            "while distant ones are more distinct in content.",
            style={"marginTop": "10px"}
        ),
        html.Details([
            html.Summary("Show Code"),
            dcc.Markdown("""
                ```python
                from bertopic import BERTopic
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
                print("\\n=== Topics ===")
                print(topic_info)

                # Optionally show a few rows
                print(reviews_sampled[['clean_review', 'topic']].head())
                         
                topic_model.visualize_topics().show()
                ```
            """)
        ], style={"marginTop": "10px"})
    ], style={
        "padding": "0 5%",
        "marginBottom": "40px"
    }),

    html.Div([
        html.Iframe(
            src="https://storage.googleapis.com/cs163-project-452620.appspot.com/analysis/topic_word_scores_fig.html",
            style={
                "width": "100%",
                "height": "600px",
                "border": "none",
                "overflow": "hidden"
            }
        ),
        html.P(
            "The topic word scores chart highlights the top keywords associated with each discovered topic, "
            "with bar lengths indicating their relative importance. This helps interpret the themes captured by the "
            "model—for example, Topic 0 is centered around pizza-related terms, while Topic 3 emphasizes sushi and "
            "Japanese cuisine. Specifically, it's worth noting in Topic 4 that 'wifi' is grouped along with words that relate "
            "to coffee shops and drinks although these words do not have the same meaning, they are still grouped in to the same "
            "Topic. This could be because many people especially students go to cafes for the wifi and caffeinated drinks "
            "to work on their assignments. BERTopic being able to capture this trend really highlights its expressiveness as "
            "a deep learning and clustering model. Moreover, we will discuss later how this is a problem for our project.",
            style={"marginTop": "10px"}
        ),
    ], style={
        "padding": "0 5%",
        "marginBottom": "40px"
    }),
    html.Details([
        html.Summary("Show Code"),
        dcc.Markdown("""
        ```python
        from bertopic import BERTopic
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
                print("\\n=== Topics ===")
                print(topic_info)

                # Optionally show a few rows
                print(reviews_sampled[['clean_review', 'topic']].head())
                         
                topic_word_scores_fig = topic_model_load.visualize_barchart()
                topic_word_scores_fig.write_html("/content/drive/MyDrive/yelp-dataset/analysis/topic_word_scores_fig.html")

                topic_word_scores_fig.show()        
        ```
        """)
    ], style={"padding": "0 5%", "marginBottom": "30px"}),

    html.H3("Topics Hierarchy", style={
        "color": "#a94442",
        "fontWeight": 800,
        "fontSize": "1.5rem",
        "borderLeft": "6px solid #a94442",
        "paddingLeft": "1rem",
        "marginTop": "30px"
    }),
    html.Div([
        html.Iframe(
            src="https://storage.googleapis.com/cs163-project-452620.appspot.com/analysis/heirarchy_tree_fig.html",
            style={
                "width": "100%",
                "height": "600px",
                "border": "none",
                "overflow": "hidden"
            }
        ),
        html.Details([
            html.Summary("Show code"),
            dcc.Markdown("""
            ```python
            topic_model.visualize_hierarchy().show()
            ```
            """)
        ], style={"marginTop": "10px"})
    ], style={
        "padding": "0 5%",
        "marginBottom": "40px"
    }),

    html.H3("BERTopic Insights:", style={
        "color": "#a94442",
        "fontWeight": 800,
        "fontSize": "1.3rem",
        "marginTop": "20px"
    }),
    html.P(
        "Due to nature of BERTopic's deep learning approach to embedding documents, it can observed in the above topic clustering "
        "and Hierarchical representations that the model is powerful and extremely expressive; however, it has untended effects that "
        "we believe is not useful for our project objective. We simply want the model to find topics such as food, service, ambience, etc., "
        "but instead we noticed that BERTopic is too fine-grained and resorts to classifying the restaurants into categories of food "
        "such as tacos, pasta, pad thai. It's as if the model is sorting reviews into categories of food, which is we believe is not "
        "useful for our analysis.",
        style={"padding": "0 5%", "marginBottom": "40px"}
    ),


    # NMF Section
    html.H2("Non-negative Matrix Factorization (NMF) Topic Modeling", style={
        "fontWeight": 900,
        "color": "#a94442",
        "fontSize": "2rem",
        "fontFamily": "'Poppins', sans-serif",
        "letterSpacing": "1px",
        "marginTop": "40px",
        "borderLeft": "6px solid #a94442",
        "paddingLeft": "1rem"
    }),
    html.P(
        "NMF is a linear topic modeling technique that decomposes document-term matrix into two "
        "smaller non-negative matrices representing topics and their word contributions. "
        "Each topic is a weighted combination of words, and each document is a mix of topics. NMF is often considered more "
        "interpretable than BERTopic because it produces sparse, additive results—making it easy to see which specific words "
        "contribute most to each topic. This simplicity allows for clearer insights, especially when the goal is to extract "
        "straightforward, human-readable themes from text data.",
        style={"padding": "0 5%"}
    ),
    html.P(
        "Below we have an NMF model with varying number of topics identified from the Yelp Dataset reviews and their "
        "top 10 words of each topic are displayed:",
        style={"padding": "0 5%", "marginBottom": "20px"}
    ),
    html.Div([
        html.Div([
            html.Label("Select Topic Count:", style={"fontWeight": "bold", "fontSize": "16px"}),
            dcc.RadioItems(
                id="nmf-topic-radio",
                options=[
                    {"label": "5", "value": 5},
                    {"label": "7", "value": 7},
                    {"label": "10", "value": 10},
                    {"label": "20", "value": 20}
                ],
                value=5,
                labelStyle={"display": "inline-block", "margin-right": "15px"},
                inputStyle={"margin-right": "6px"},
                style={"margin-bottom": "10px"},
            ),
            dash_table.DataTable(
                id="nmf-topics",
                columns=[{"name": "Topic", "id": "Topic"}, {"name": "Top Words", "id": "Words"}],
                data=[],
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "padding": "5px"},
                style_header={"backgroundColor": "#f2f2f2", "fontWeight": "bold"},
            ),
        ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top"}),

        html.Div([
            dcc.Graph(id='nmf-topics-pie')
        ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top"}),
    ], style={"padding": "0 5%", "marginBottom": "40px"}),

    html.Details([
        html.Summary("Show Code"),
        dcc.Markdown("""
    ```python
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
        ```
        """)
    ], style={"padding": "0 5%", "marginBottom": "30px"}),

    

    # Correlation Section
    html.H2("Correlation Between Open Status and Average Rating", style={
        "fontWeight": 900,
        "color": "#a94442",
        "fontSize": "2rem",
        "fontFamily": "'Poppins', sans-serif",
        "letterSpacing": "1px",
        "marginTop": "40px",
        "borderLeft": "6px solid #a94442",
        "paddingLeft": "1rem"
    }),
    html.P(
        "We calculated the Pearson corelation value between open status and average rating was got a value of 0.054492. "
        "Since the value was statistically insignificant, we conducted a t-test. After conducting a t-test, we obtained a p-value of 0, which seemed odd. After digging deeper, "
        "we realized this was due to the large number of columns in the dataset. To handle this, we used a random sampled dataset of a 1000 rows  "
        "and obtained a p-value of 0.037, which is less than an alpha value of 0.05. "
        "This indicated that the correlation value was statistically significant.",
        style={"padding": "0 5%", "marginBottom": "40px"}
    ),
    html.Details([
        html.Summary("Show Code"),
        dcc.Markdown("""
        ```python
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

        ```
        """)
    ], style={"padding": "0 5%", "marginBottom": "30px"}),

    # Key Insights Section
    html.H3("Key Insights", style={
        "marginTop": "40px",
        "color": "#a94442",
        "fontWeight": 800,
        "fontSize": "1.5rem",
        "borderLeft": "6px solid #a94442",
        "paddingLeft": "1rem"
    }),
    html.Ul([
        html.Li([
            "Adjectives with a ",
            html.Strong("positive connotation"),
            " tend to correlate with ",
            html.Strong("higher ratings"),
            ", while those with a ",
            html.Strong("negative connotation"),
            " tend to correlate with ",
            html.Strong("lower ratings"),
            "."
        ]),
        html.Li([
            "Even though BERTopic was comprehensive, its ",
            html.Strong("fine-grained classification"),
            " wasn't ideal for identifying broader topics like ",
            html.Strong("food"),
            ", ",
            html.Strong("service"),
            ", and ",
            html.Strong("ambience"),
            "."
        ]),
        html.Li([
            "NMF topic modeling provides a more ",
            html.Strong("generalized view"),
            " of topics, making it better suited."
        ])
    ], style={
        "backgroundColor": "#f8f8f8",
        "padding": "20px",
        "fontFamily": "Poppins, sans-serif",
        "lineHeight": "1.8",
        "marginTop": "20px",
        "borderLeft": "4px solid #d9534f",
        "marginBottom": "40px"
    }),

], 
style={"fontFamily": "Poppins, sans-serif", "padding": "20px", "backgroundColor": "#fff"})
