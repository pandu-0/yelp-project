import dash
from dash import html, dcc
import plotly.express as px  # keep this for your real visualizations

dash.register_page(__name__, path='/eda', name='EDA')

layout = html.Div([
    html.H2("Exploratory Data Analysis", style={"fontWeight": "600"}),

    html.H3("Dataset Summary"),
    html.Ul([
        html.Li("Contains nearly 65,000 restaurants"),
        html.Li("Median number of reviews per restaurant: 15"),
        html.Li("Data types: Numerical, Categorical, and Text"),
        html.Li("Features: Review text, review rating, average rating, categories"),
        html.Li("Target: Restaurant (for clustering by topic/sentiment modeling)")
    ]),

    html.H3("Preprocessing"),
    html.Ul([
        html.Li("103 null values found only in the Categories column"),
        html.Li("9 duplicate rows detected and removed"),
        html.Li("No other missing values")
    ]),

    html.H3("Numerical Features"),
    html.Ul([
        html.Li("review_stars: rating left by a reviewer (out of 5)"),
        html.Li("avg_review: average star rating for a restaurant (out of 5)"),
        html.Li("review_count: number of reviews per restaurant"),
        html.Li("is_open: binary indicator of whether a restaurant is still open")
    ]),

    html.H3("Visualizations"),
    html.P("Below are placeholders for the EDA graphs. Replace these with your actual Plotly figures."),

    # html.Div([
    #     html.H4("Review Stars Distribution"),
    #     dcc.Graph(id='review-stars-dist', figure=px.histogram()),  # Replace with actual data
    # ], style={"marginBottom": "40px"}),

    # html.Div([
    #     html.H4("Average Rating Histogram"),
    #     dcc.Graph(id='avg-review-hist', figure=px.histogram()),  # Replace with actual data
    # ], style={"marginBottom": "40px"}),

    # html.Div([
    #     html.H4("Review Count Distribution (Outliers Removed)"),
    #     dcc.Graph(id='review-count-dist', figure=px.histogram()),  # Replace with actual data
    # ], style={"marginBottom": "40px"}),

    # html.Div([
    #     html.H4("Open vs Closed Restaurants"),
    #     dcc.Graph(id='is-open-pie', figure=px.pie()),  # Replace with actual data
    # ]),

    html.H3("Preliminary Insights"),
    html.Ul([
        html.Li("5-star ratings are the most commonly given"),
        html.Li("The most common average rating is 4 stars"),
        html.Li("Majority of restaurants are still open")
    ]),

    html.H3("Next Steps & Hypotheses"),
    html.Ul([
        html.Li("Sentiment analysis may impact ratings"),
        html.Li("Certain keywords may correlate with high or low ratings")
    ]),

    html.Br(),
    dcc.Link(html.Button("Back to Home"), href="/")
], style={"fontFamily": "Poppins, sans-serif", "padding": "20px"})

