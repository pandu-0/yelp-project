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
    # Header Section
    html.Div([
        html.H2("🍽️ Restaurant Review Analysis", style={
            "fontWeight": 900,
            "marginBottom": "8px",
            "color": "#a94442",
            "fontSize": "2.7rem",
            "fontFamily": "'Poppins', sans-serif",
            "letterSpacing": "1.5px",
            "textShadow": "1px 2px 8px #fbeee6"
        }),
        html.P("Group ID: 19", style={"marginBottom": "2px", "color": "#555", "fontSize": "1.08rem"}),
        html.P("Members: Pandu Rupanagudi, Singaram Subramanyan", style={"marginBottom": "0", "color": "#555", "fontSize": "1.08rem"}),
    ], style={
        "background": "linear-gradient(90deg, #fbeee6 0%, #f7cac9 100%)",
        "padding": "2.2rem 2.7rem 1.2rem 2.7rem",
        "borderRadius": "1.2rem",
        "boxShadow": "0 4px 18px rgba(184,92,56,0.13)",
        "marginBottom": "2.2rem",
        "border": "2px solid #e0e0e0"
    }),

    # Project Summary
    html.Div([
        html.H3("🥗 Project Summary", style={
            "color": "#a94442",
            "marginTop": 0,
            "fontWeight": 800,
            "borderLeft": "6px solid #a94442",
            "paddingLeft": "1rem",
            "fontSize": "1.5rem"
        }),
        html.P(
            "This project analyzes restaurant customer reviews to quantify quality using natural language processing. "
            "We uncover satisfaction patterns and key concerns via sentiment analysis and topic modeling, "
            "using review scores and polarity across Yelp and Google datasets. "
            "Our insights benefit both customers and businesses.",
            style={"color": "#333", "fontSize": "1.13rem", "marginTop": "0.7rem"}
        ),
    ], style={"marginBottom": "1.7rem", "background": "#fff", "borderRadius": "0.9rem", "padding": "1.2rem 2rem", "boxShadow": "0 2px 8px #fbeee633"}),

    # Broader Impacts
    html.Div([
        html.H3("🍕 Broader Impacts", style={
            "color": "#a94442",
            "fontWeight": 800,
            "borderLeft": "6px solid #a94442",
            "paddingLeft": "1rem",
            "fontSize": "1.5rem"
        }),
        html.Ul([
            html.Li("Enhanced Consumer Experience: Better dining choices based on review trends."),
            html.Li("Increased Restaurant Performance: Actionable insights to improve satisfaction."),
            html.Li("Informed Decision Making: Address complaints and improve service using data."),
        ], style={"marginLeft": "2rem", "lineHeight": "1.8", "color": "#444", "fontSize": "1.08rem"})
    ], style={"marginBottom": "1.7rem", "background": "#fff", "borderRadius": "0.9rem", "padding": "1.2rem 2rem", "boxShadow": "0 2px 8px #fbeee633"}),

    # Data Sources
    html.Div([
        html.H3("🥑 Data Sources", style={
            "color": "#a94442",
            "fontWeight": 800,
            "borderLeft": "6px solid #a94442",
            "paddingLeft": "1rem",
            "fontSize": "1.5rem"
        }),
        html.Ul([
            html.Li(html.A("Yelp Open Dataset", href="https://business.yelp.com/data/resources/open-dataset/", target="_blank", style={"color": "#2d7f5e", "fontWeight": 600})),
            html.Li(html.A("Google Restaurant Dataset via Julian McAuley", href="https://cseweb.ucsd.edu/~jmcauley/datasets.html#google_restaurants", target="_blank", style={"color": "#2d7f5e", "fontWeight": 600})),
            html.Li(html.A("An Yan et al. (2022), Multi-Modal Explanations for Recommendations", href="https://arxiv.org/abs/2207.00422", target="_blank", style={"color": "#2d7f5e", "fontWeight": 600}))
        ], style={"marginLeft": "2rem", "lineHeight": "1.8", "color": "#444", "fontSize": "1.08rem"})
    ], style={"marginBottom": "2.2rem", "background": "#fff", "borderRadius": "0.9rem", "padding": "1.2rem 2rem", "boxShadow": "0 2px 8px #fbeee633"}),

    # NMF Topic Modeling Analysis
    html.Div([
        html.H3("🍜 NMF Topic Modeling Analysis (7 Topics)", style={
            "color": "#a94442",
            "fontWeight": 800,
            "borderLeft": "6px solid #a94442",
            "paddingLeft": "1rem",
            "fontSize": "1.5rem"
        }),
        html.P(
            "We performed topic modeling with NMF to identify 7 prevalent topics in the Yelp Review dataset. "
            "These topics enabled the creation of the graphs below, which show that Philadelphia's restaurants perform poorly "
            "in timely service. The second plot shows a decline in mean stars for Topic 1 (timely service). "
            "This information can help businesses improve their service.",
            style={"color": "#333", "fontSize": "1.13rem", "marginTop": "0.7rem"}
        ),
        # Topic Table
        dash_table.DataTable(
            columns=[
                {"name": "Topic", "id": "Topic"},
                {"name": "Top Words", "id": "Words"}
            ],
            data=[
                {"Topic": f"Topic {i}", "Words": kw}
                for i, kw in enumerate(topic_keywords)
            ],
            style_table={"overflowX": "auto", "borderRadius": "0.8rem", "background": "#fff", "marginTop": "1.2rem"},
            style_cell={
                "textAlign": "left",
                "padding": "0.8rem",
                "fontFamily": "'Poppins', 'sans-serif'",
                "fontSize": "1.07rem",
                "backgroundColor": "#f9f6f2",
                "color": "#3d3d3d"
            },
            style_header={
                "backgroundColor": "#fbeee6",
                "fontWeight": "bold",
                "fontSize": "1.12rem",
                "borderBottom": "2px solid #a94442",
                "color": "#a94442"
            },
        ),
        # First Plot
        html.Div([
            html.Img(
                src="https://raw.githubusercontent.com/pandu-0/yelp-project/main/assets/mean_rating_over_topic_7_open_status_plot.svg",
                style={
                    'maxWidth': '100%',
                    'height': 'auto',
                    "borderRadius": "0.8rem",
                    "boxShadow": "0 2px 12px rgba(184,92,56,0.13)",
                    "marginTop": "1.5rem",
                    "border": "2px solid #f7cac9"
                }
            )
        ]),
        # Second Plot
        html.Div([
            html.Img(
                src="https://raw.githubusercontent.com/pandu-0/yelp-project/main/assets/mean_rating_over_year_topic_7_plot.svg",
                style={
                    'width': '100%',
                    'height': 'auto',
                    "borderRadius": "0.8rem",
                    "boxShadow": "0 2px 12px rgba(184,92,56,0.13)",
                    "marginTop": "1.5rem",
                    "border": "2px solid #f7cac9"
                }
            )
        ]),
    ], style={
        "background": "linear-gradient(90deg, #fbeee6 0%, #f7cac9 100%)",
        "padding": "2.2rem 2.7rem",
        "borderRadius": "1.2rem",
        "boxShadow": "0 4px 18px rgba(184,92,56,0.10)",
        "marginBottom": "2.2rem",
        "border": "2px solid #e0e0e0"
    }),

    # Major Findings
    html.Div([
        html.H4("🍰 Major Findings", style={
            "color": "#a94442",
            "marginTop": 0,
            "fontWeight": 800,
            "borderLeft": "6px solid #a94442",
            "paddingLeft": "1rem",
            "fontSize": "1.25rem"
        }),
        html.Ul([
            html.Li(
                "There is no statistically significant difference in mean star ratings of open and closed restaurants across any topic or over time. "
                "This suggests that closures may be due to other factors, such as competitiveness, as most restaurants are centrally located."
            ),
            html.Li(
                "The lower mean review star of Topic 1 indicates that restaurants in Philadelphia are performing poorly in terms of timely service."
            )
        ], style={"marginLeft": "2rem", "lineHeight": "1.8", "color": "#444", "fontSize": "1.08rem"})
    ], style={
        "backgroundColor": "#f9f6f2",
        "padding": "1.3rem 2.2rem",
        "fontFamily": "'Poppins', 'sans-serif'",
        "lineHeight": "1.9",
        "borderLeft": "7px solid #a94442",
        "borderRadius": "0.9rem",
        "marginBottom": "1.5rem",
        "boxShadow": "0 2px 8px #fbeee633"
    })
], style={
    "fontFamily": "'Poppins', 'sans-serif'",
    "padding": "2.7rem 0",
    "background": "#f6f8fa",
    "minHeight": "100vh"
})
