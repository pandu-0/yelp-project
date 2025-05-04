from dash import html, dcc
import dash

dash.register_page(__name__, path='/analysis', name='Analysis')

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

    html.Br(),
    dcc.Link(html.Button("Back to Home"), href="/"),

], style={"fontFamily": "Poppins, sans-serif", "padding": "20px"})
