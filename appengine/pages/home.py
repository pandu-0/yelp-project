import dash
from dash import html, dash_table
from utils import GCLOUD_BUCKET, load_pickle_from_gcs

dash.register_page(__name__, path='/', name='Home')

# load vectorizer
vectorizer = load_pickle_from_gcs(GCLOUD_BUCKET, "models/nmf_model/tfidf_vectorizer.pkl")

# Load the saved mode
nmf_model = load_pickle_from_gcs(GCLOUD_BUCKET, f"models/nmf_model/nmf_model_topics_7.pkl")

# Extract feature names (vocabulary words)
feature_names = vectorizer.get_feature_names_out()

# Show top N words per topic
no_top_words = 7
nmf_topics = [
    [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
    for topic in nmf_model.components_
]

topic_keywords = [" | ".join(words) for words in nmf_topics]


layout = html.Div([
    html.H2("Restaurant Review Analysis", style={"fontWeight": "600", "marginBottom": "10px"}),
    html.P("Group ID: 19"),
    html.P("Members: Pandu Rupanagudi, Singaram Subramanyan"),

    html.H3("Project Summary"),
    html.P(
        "The aim of this project is to analyze customer reviews of restaurants and measure quality in a quantifiable manner. "
        "Using natural language processing (e.g., sentiment analysis), we examine reviews to uncover satisfaction patterns and identify key concerns. "
        "We use average review scores, sentiment polarity, and topic modeling across Yelp and Google restaurant datasets to provide insights for both customers and businesses."
    ),

    html.H3("Broader Impacts"),
    html.Ul([
        html.Li("Enhanced Consumer Experience: Helps customers make better dining choices based on review trends."),
        html.Li("Increased Restaurant Performance: Provides actionable insights for restaurants to improve customer satisfaction."),
        html.Li("Informed Decision Making: Allows restaurants to address complaints and improve service using data.")
    ]),

    html.H3("Data Sources"),
    html.Ul([
        html.Li([
            html.A("Yelp Open Dataset", href="https://business.yelp.com/data/resources/open-dataset/", target="_blank")
        ]),
        html.Li([
            html.A("Google Restaurant Dataset via Julian McAuley", href="https://cseweb.ucsd.edu/~jmcauley/datasets.html#google_restaurants", target="_blank")
        ]),
        html.Li([
            html.A("An Yan et al. (2022), Multi-Modal Explanations for Recommendations", href="https://arxiv.org/abs/2207.00422", target="_blank")
        ])
    ]), 

    html.H3("NMF Topics Modeling Analysis (7 Topics)"),
    html.P(
        "We performed Topic modeling with NMF to identify 7 prevalent topics in the Yelp Review dataset. These topics allowed us " \
        "to create the graphs shown below, which demonstrate that Philadelphia's restaurants perform poorly in " \
        "terms of timely service to customers that place an order.  Moreover, it can also be noticed in the second plot that "
        "the mean stars of restaurants in Topic 1 (timely service) has been on the decline. " \
        "This information can help businesses improve their service to maximize customer satisfaction."
    ),
    
    html.Div([
        html.Div([
            dash_table.DataTable(
                columns=[{"name": "Topic", "id": "Topic"}, {"name": "Top Words", "id": "Words"}],
                data=[{"Topic": f"Topic {i}", "Words": kw} for i, kw in enumerate(topic_keywords)],
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "padding": "5px"},
                style_header={"backgroundColor": "#f2f2f2", "fontWeight": "bold"},
            )
        ], style={
            "overflowX": "auto",
            "whiteSpace": "nowrap",
            "width": "100%",
            "paddingBottom": "20px"
        }),
        html.Div([
            html.Img(
                src="https://raw.githubusercontent.com/pandu-0/yelp-project/main/assets/mean_rating_over_topic_7_open_status_plot.svg",
                style={'maxWidth': '100%', 'height': 'auto'}
            )
        ], style={
            "overflowX": "auto",
            "whiteSpace": "nowrap",
            "width": "100%",
            "paddingBottom": "20px"
        }),
    ], style={
        "display": "flex",
        "padding": "0 5%",
        "gap": "20px",
        "marginBottom": "40px"
    }),

    html.Div([
    html.Img(
        src="https://raw.githubusercontent.com/pandu-0/yelp-project/main/assets/mean_rating_over_year_topic_7_plot.svg",
        style={'width': '100%', 'height': 'auto'}
    )
    ], style={"width": "100%", "display": "inline-block", "verticalAlign": "top"}),

    html.Div([
    html.H4("Major Findings"),
    html.Ul([
        html.Li([
            "There is no statistically significant difference in mean star ratings of open and closed restaurants across any topic"
            "or over the course of time. "
            "This suggests that the reason for restaurant closures might be due to other circumstances such as "
            "competitiveness (no way to measure in our dataset) as most "
            "restaurant are located in the center of the city"
        ]),
        html.Li([
            "The lower mean review star of Topic 1 indicates that restaurants in Philadelphia are performing poorly in terms "
            "of timely service of orders."
        ])
    ])
    ], style={
        "backgroundColor": "#f8f8f8",
        "padding": "20px",
        "fontFamily": "Poppins, sans-serif",
        "lineHeight": "1.8",
        "marginTop": "20px",
        "borderLeft": "4px solid #d9534f",
        "marginBottom" : "20px"
    })

], style={"fontFamily": "Poppins, sans-serif", "padding": "20px"})
