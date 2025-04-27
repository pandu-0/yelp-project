import dash
from dash import html

dash.register_page(__name__, path='/', name='Home')

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
    ])
], style={"fontFamily": "Poppins, sans-serif", "padding": "20px"})
