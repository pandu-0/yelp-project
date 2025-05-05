import dash
from dash import html, dcc, dash_table
import pandas as pd
import plotly.express as px
from utils import get_json_from_gcs, load_pickle_from_gcs, GCLOUD_BUCKET

dash.register_page(__name__, path='/analysis-methods', name='Analysis')

df = pd.read_csv("https://raw.githubusercontent.com/pandu-0/yelp-project/refs/heads/main/preview_datasets/Adjective_correlation_rating")

df["Correlation"] = df["Correlation"].round(2)
df["average_rating"] = df["average_rating"].round(2)

top_5 = df.sort_values(by="Correlation", ascending=False).head(5)
bottom_5 = df.sort_values(by="Correlation", ascending=False).tail(5)

table_rows = [
    html.Tr([html.Td(row["Adjective"]), html.Td(row["Correlation"]), html.Td(row["average_rating"])])
    for i, row in pd.concat([top_5, bottom_5]).iterrows()
]

# Load the saved model and vectorizer
nmf_model = load_pickle_from_gcs(GCLOUD_BUCKET, "models/nmf_model/nmf_model.pkl")
vectorizer = load_pickle_from_gcs(GCLOUD_BUCKET, "models/nmf_model/tfidf_vectorizer.pkl")

# Extract feature names (vocabulary words)
feature_names = vectorizer.get_feature_names_out()

# Show top N words per topic
no_top_words = 10
nmf_topics = [
    [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
    for topic in nmf_model.components_
]

topic_keywords = [" | ".join(words) for words in nmf_topics]

# retrieve topics
philly_nmf_topics = get_json_from_gcs(GCLOUD_BUCKET, "models/nmf_model/philly_balanced_with_sentiment_nmf_topics.json")

# Pie chart
topic_counts = philly_nmf_topics['topic'].value_counts().sort_index()

layout = html.Div([
    html.H1("Analysis Methods", style={"fontWeight": "600"}),

    # description about the analysis
    html.H3("Adjective Correlation with Ratings", style={"marginTop": "30px"}),
    html.Div([
        html.P(
            "This analysis aimed to determine if there was a correlation between the presence of certain adjectives "
            "and the review rating. "
            "This was accomplished through using the nltk to tokenize each review and filtering for adjectives using " \
            "POS (Part of Speech) tagging. "
            "Once filtered down to adjectives, the Pearson correlation coefficient between adjectives and the rating was calculated. "
            "The table below depicts the 5 most and least correlated adjectives."
        )
    ], style={
        "padding": "0 5%",
        "marginBottom": "20px"
    }),

    # table
    html.Div([
        html.Table([
            html.Thead(
                html.Tr([
                    html.Th("Adjective"),
                    html.Th("Correlation Coefficient"),
                    html.Th("Average Rating")
                ])
            ),
            html.Tbody(table_rows)
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
            "Moreover, positively correlated adjectives tend to be adjectives with a positive connotation, such as 'great', " \
            "'good', and 'amazing,' while the opposite is true for adjectives with a negative connotation. "
            "This indicates that a the presence of a positive connotation adjective increases the chances of receiving a " \
            "higher rating, while the presence of a negative connotation adjective increases the chances of " \
            "receiving a lower rating. "
        )
    ], style={
        "padding": "0 5%",
        "marginBottom": "40px"
    }),

    # Preprocessing
    html.H2("Preprocessing", style={"fontWeight": "600"}),
    html.Div([
        html.Ul([
            html.Li("The Yelp dataset was mostly clean with minimal null values and duplicates."),
            html.Li("There was a class imbalance between open and closed restaurants."),
            html.Li("To address this, open restaurants were under-sampled to match the number of closed ones."),
            html.Li("Text data was cleaned by removing stop words, punctuation, and applying lemmatization."),
            html.Li("Manual cleaning was chosen over TF-IDF to reduce dataset size."),
            html.Li("We noticed cleaning helped improve the clarity of the topics identified by the various models.")
        ]),
        # show the preprocessing code
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
        ])
    ], style={
        "padding": "0 5%",
        "marginBottom": "20px"
    }),

    html.H2("Topic Modeling", style={"fontWeight": "600"}),

    html.H3("BERTopic Analysis", style={"fontWeight": "600"}),
    # topics figure
    html.H4("Topics Distance Map", style={"marginTop": "30px"}),
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

    # topics word scores figure
    # html.H4("Topic word scores figure", style={"marginTop": "30px"}),
    # html.Div([
    #     html.Iframe(
    #         src="https://storage.googleapis.com/cs163-project-452620.appspot.com/analysis/topic_word_scores_fig.html",
    #         style={
    #             "width": "100%",
    #             "height": "600px",
    #             "border": "none",
    #             "overflow": "hidden"
    #         }
    #     )
    # ], style={
    #     "padding": "0 5%",
    #     "marginBottom": "40px"
    # }),

    # hierarchy tree figure
    html.H4("Topics Hierarchy ", style={"marginTop": "30px"}),
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

    html.H5("BERTopic Insights:"),
    html.P(
        "Due to nature of BERTopic's deep learning approach to embedding documents, it can observed in the above topic clustering " \
        "and Hierarchical representations that the model is powerful and extremely expressive; however, it has untended effects that " \
        "we believe is not useful for our project objective. We simply want the model to find topics such as food, service, ambience, etc., " \
        "but instead we noticed that BERTopic is too fine-grained and resorts to classifying the restaurants into categories of food " \
        "such as tacos, pasta, pad thai. It's as if the model is sorting reviews into categories of food, which is we believe is not " \
        "useful for our analysis."
    ),

    html.H2("NMF Topic Modeling", style={"fontWeight": "600"}),

    html.Div([
        html.Div([
            html.H4("Top Keywords per Topic"),
            dash_table.DataTable(
                columns=[{"name": "Topic", "id": "Topic"}, {"name": "Top Words", "id": "Words"}],
                data=[{"Topic": f"Topic {i}", "Words": kw} for i, kw in enumerate(topic_keywords)],
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "padding": "5px"},
                style_header={"backgroundColor": "#f2f2f2", "fontWeight": "bold"},
            )
        ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top"}),

        html.Div([
            html.H4("Topic Distribution"),
            dcc.Graph(figure=px.pie(
                values=topic_counts.values,
                names=[f"Topic {i}" for i in topic_counts.index],
                title=None,
                hole=0.3
            ))
        ], style={"width": "48%", "display": "inline-block", "paddingLeft": "2%"})
    ], style={"padding": "0 5%", "marginBottom": "40px"}),

    html.Br(),
    dcc.Link(html.Button("Back to Home"), href="/"),

], style={"fontFamily": "Poppins, sans-serif", "padding": "20px"})
