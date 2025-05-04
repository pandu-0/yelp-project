from dash import html, dcc
import dash
import pandas as pd

dash.register_page(__name__, path='/analysis', name='Analysis')

df = pd.read_csv("https://raw.githubusercontent.com/pandu-0/yelp-project/refs/heads/main/preview_datasets/Adjective_correlation_rating")

df["Correlation"] = df["Correlation"].round(2)
df["average_rating"] = df["average_rating"].round(2)

top_5 = df.sort_values(by="Correlation", ascending=False).head(5)
bottom_5 = df.sort_values(by="Correlation", ascending=False).tail(5)

table_rows = [
    html.Tr([html.Td(row["Adjective"]), html.Td(row["Correlation"]), html.Td(row["average_rating"])])
    for i, row in pd.concat([top_5, bottom_5]).iterrows()
]

layout = html.Div([
    html.H1("Analysis", style={"fontWeight": "600"}),

    html.H2("BERTopic Analysis", style={"fontWeight": "600"}),
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
        )
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
        )
    ], style={
        "padding": "0 5%",
        "marginBottom": "40px"
    }),

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
    html.Br(),
    dcc.Link(html.Button("Back to Home"), href="/"),

], style={"fontFamily": "Poppins, sans-serif", "padding": "20px"})
