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

    html.H2("Sentiment Analysis", style={"fontWeight": "600", "textAlign": "center"}),
    html.Div([
        html.H3("Sentiment Probability by Star Rating"),
        html.Img(
            src="https://raw.githubusercontent.com/pandu-0/yelp-project/refs/heads/main/assets/sentiment_plot.svg",
            style={'width': '100%', 'height': 'auto'}
        )
    ]),

    html.P(
        "The above plot shows the distribution of sentiment probabilities for each star rating. "
        "We can see that the model works because it assigns higher positivity probability to high "
        "ratings and lower positivity probability to low ratings. "
        "This is a good sign that the model is working as intended. "),

    html.H2("Topic Modeling", style={"fontWeight": "600", "textAlign": "center"}),

    html.H3("BERTopic Analysis", style={"fontWeight": "600"}),
    # topics figure
    html.H3("Topics Distance Map", style={"marginTop": "30px"}),
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
    html.H3("Topic word scores figure", style={"marginTop": "30px"}),
    html.Div([
        html.Iframe(
            src="https://storage.googleapis.com/cs163-project-452620.appspot.com/analysis/topic_word_scores_fig.html",
            style={
                "width": "100%",
                "height": "600px",
                "border": "none",
                "overflow": "hidden"
            }
        )
    ], style={
        "padding": "0 5%",
        "marginBottom": "40px"
    }),

    # hierarchy tree figure
    html.H3("Topics Hierarchy ", style={"marginTop": "30px"}),
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

    html.H3("BERTopic Insights:"),
    html.P(
        "Due to nature of BERTopic's deep learning approach to embedding documents, it can observed in the above topic clustering " \
        "and Hierarchical representations that the model is powerful and extremely expressive; however, it has untended effects that " \
        "we believe is not useful for our project objective. We simply want the model to find topics such as food, service, ambience, etc., " \
        "but instead we noticed that BERTopic is too fine-grained and resorts to classifying the restaurants into categories of food " \
        "such as tacos, pasta, pad thai. It's as if the model is sorting reviews into categories of food, which is we believe is not " \
        "useful for our analysis."
    ),

    html.H2("Non-negative Matrix Factorization (NMF) Topic Modeling", style={"fontWeight": "600"}),
    html.Div([
        html.Div([
            html.Label("Select Topic Count:", style={"fontWeight": "bold", "fontSize": "16px"}),
            dcc.RadioItems(
                id="nmf-topic-radio",
                options=[
                    {"label": "5", "value": 5},
                    {"label": "7", "value": 7},
                    {"label": "10", "value": 10},
                    {"label": "20", "value" : 20}
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
            html.P("The map shows the restaurants in Philadelphia by open status, star rating, and review count. "
                   "By hovering over the dots, more information, such as the name of the restaurant and category, can be seen. ", 
                   style={"marginTop": "10px", "fontSize": "17px"})
        ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top"}),

        html.Div([
            dcc.Graph(id='nmf-topics-pie')
        ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top"})
    ], style={"padding": "0 5%", "marginBottom": "40px"}),

    html.Br(),
    dcc.Link(html.Button("Back to Home"), href="/"),

    # Key Insights Section
    html.H3("Key Insights", style={"marginTop": "40px"}),
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
        "borderRadius": "10px",
        "fontFamily": "Poppins, sans-serif",
        "lineHeight": "1.8",
        "marginTop": "20px"
    }),

], style={"fontFamily": "Poppins, sans-serif", "padding": "20px"})
