from dash import html, dcc, dash_table
from dash.dependencies import Input, Output
import dash
import pandas as pd
import plotly.express as px
from utils import load_pickle_from_gcs, GCLOUD_BUCKET

dash.register_page(__name__, path='/major-findings', name='Major-findings')
df = pd.read_csv('https://raw.githubusercontent.com/pandu-0/yelp-project/refs/heads/main/assets/Adjective_correlation_rating')

# Sort top 5 and bottom 5
top5 = df.sort_values(by='Correlation', ascending=False).head(5)
bottom5 = df.sort_values(by='Correlation', ascending=True).head(5)

@dash.callback(
    Output('correlation-graph', 'figure'),
    Input('group-selector', 'value')
)
def update_graph(selected_group):
    if selected_group == 'top5':
        filtered_df = top5
    elif selected_group == 'bottom5':
        filtered_df = bottom5
    else:
        filtered_df = pd.concat([top5, bottom5])

    fig = px.bar(
        filtered_df,
        x='Correlation',    
        y='Adjective',      
        orientation='h',     
        color='Correlation',
        color_continuous_scale='RdBu',
        hover_data={'average_rating': True, 'Correlation': ':.2f'},
        labels={'Correlation': 'Correlation with Rating', 'Adjective': 'Adjective'},
        title='Top 5 and Bottom 5 Correlated Adjectives'
    )

    fig.update_layout(
        xaxis_title="Correlation",
        yaxis_title="Adjective",
        coloraxis_colorbar=dict(title="Correlation"),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Poppins"),
        yaxis=dict(autorange="reversed")  # Optional: reverse so top items appear at the top
    )

    return fig

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

# App layout
layout = html.Div([
    html.H2("Major Findings and Visualization"),

    html.Div([
        html.Label("Select Adjective Group:"),
        dcc.RadioItems(
            id='group-selector',
            options=[
                {'label': 'Top 5', 'value': 'top5'},
                {'label': 'Bottom 5', 'value': 'bottom5'},
                {'label': 'Both', 'value': 'both'}
            ],
            value='both',  # default selected
            inline=True,
            style={"marginBottom": "20px"}
        ),

        dcc.Graph(id='correlation-graph'),
    ]),

    html.Div([
        html.H4("Correlation Between Adjectives and Ratings"),

        html.P("The correlation between adjectives and ratings was analyzed using the Pearson correlation coefficient. " \
                "The analysis revealed that adjectives with a positive connotation tend to have a positive correlation with ratings, " \
                "while adjectives with a negative connotation tend to have a negative correlation with ratings. ",
                style={"lineHeight": "1.6", "marginBottom": "10px"}),

    ], style={
        "backgroundColor": "#f8f8f8",
        "padding": "20px",
        "fontFamily": "Poppins, sans-serif",
        "lineHeight": "1.8",
        "marginTop": "20px",
        "borderLeft": "4px solid #d9534f",
        "marginBottom" : "20px"
    }),

    html.H3("NMF Topics Modeling Analysis"),
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
        html.H4("Statistical Test: Topic 1 Ratings"),

        html.P("ANOVA (Analysis of Variance) followed by Tukey's HSD test revealed that Topic 1 has a significantly lower mean star " \
                "rating compared to all other topics (Topics 2-6), with adjusted p-values < 0.001 in all cases. " \
                "The mean differences ranged from 0.8 to 1.3 stars, indicating a substantial gap in perceived quality.",
                style={"lineHeight": "1.6", "marginBottom": "10px"}),

        html.P("Only Topic 0 had a similar rating to Topic 1, but it was still significantly different. " \
                "These findings suggest Topic 1 is strongly associated with lower-rated experiences.",
                style={"lineHeight": "1.6"}),

        html.Details([
            html.Summary("Show code"),
            dcc.Markdown("""
            ```python
            import statsmodels.api as sm
            from statsmodels.formula.api import ols
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
                                    
            test_df = philly_balanced_clean[['topic', 'stars']]
            test_df.head()

            # ANOVA
            model = ols('stars ~ C(topic)', data=test_df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            print(anova_table)

            # Post-hoc Tukey HSD
            tukey = pairwise_tukeyhsd(test_df['stars'], test_df['topic'], alpha=0.05)
            print(tukey.summary())
            ```
            """)
        ], style={"marginTop": "10px"})
    ], style={
        "backgroundColor": "#f8f8f8",
        "padding": "20px",
        "fontFamily": "Poppins, sans-serif",
        "lineHeight": "1.8",
        "marginTop": "20px",
        "borderLeft": "4px solid #d9534f",
        "marginBottom" : "20px"
    }),

    html.Div([
        html.Img(
            src="https://raw.githubusercontent.com/pandu-0/yelp-project/main/assets/mean_rating_over_year_topic_7_plot.svg",
            style={'width': '100%', 'height': 'auto'}
        )
    ], style={"width": "100%", "display": "inline-block", "verticalAlign": "top"}),

])
