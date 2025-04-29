import dash
from dash import html, dcc, dash_table
import pandas as pd
from google.cloud import storage
import os
from io import StringIO
import plotly.express as px

dash.register_page(__name__, path='/eda', name='EDA')

# --------- Load JSON files ---------
def get_json_from_gcs(bucket_name, source_blob_name):
    """Downloads a blob from the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The ID of your GCS object
    # source_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    data = blob.download_as_text()
    return pd.read_json(StringIO(data), lines=True)

# --------- Load JSON files ---------
review_df = pd.read_json('https://raw.githubusercontent.com/pandu-0/yelp-project/refs/heads/main/preview_datasets/review_head.json',
                        lines=True)
user_df = pd.read_json('https://raw.githubusercontent.com/pandu-0/yelp-project/refs/heads/main/preview_datasets/user_head.json',
                        lines=True)
tips_df = pd.read_json('https://raw.githubusercontent.com/pandu-0/yelp-project/refs/heads/main/preview_datasets/tips_head.json',
                        lines=True)

business_df = pd.read_json('https://raw.githubusercontent.com/pandu-0/yelp-project/refs/heads/main/preview_datasets/business_head.json',
                        lines=True)
# Convert nested columns to strings (if don't do this, it causes issues with displaying in DataTable)
business_df['attributes'] = business_df['attributes'].astype(str)
business_df['hours'] = business_df['hours'].astype(str)

checkin_df = pd.read_json('https://raw.githubusercontent.com/pandu-0/yelp-project/refs/heads/main/preview_datasets/checkin_head.json',
                        lines=True)

# show a few rows (head) for preview
review_preview = review_df.to_dict('records')
user_preview = user_df.to_dict('records')
tips_preview = tips_df.to_dict('records')
business_preview = business_df.to_dict('records')
checkin_preview = checkin_df.to_dict('records')

philly_reviews_df = get_json_from_gcs("cs163-project-452620.appspot.com", "philly_restaurants_reviews_for_gcloud.json")
# --------- Compute statistics dynamically ---------
unique_restaurants_count = philly_reviews_df['business_id'].nunique()

stars_summary = philly_reviews_df['stars'].describe().round(2).to_frame().reset_index()
stars_summary.columns = ['Statistic', 'Stars']  # Rename columns for nice table display


# --------- Layout ---------
layout = html.Div([
    html.H1("Exploratory Data Analysis", style={"fontWeight": "600"}),

    html.H3("Yelp Dataset Summary:", style={"marginTop": "30px"}),
    html.P(html.Span(["The Yelp Open dataset comes with 5 files in ", html.Code("json"), " format:", ])),
    html.Ul([
        html.Li(html.Span([html.Code("Review"), " : contains reviews and related data left by a customer for a business"])),
        html.Li(html.Span([html.Code("User"), " : contains the users and their profile information"])),
        html.Li(html.Span([html.Code("Tips"), " : contains tip or comment left by a user for a business"])),
        html.Li(html.Span([html.Code("Business"), " : contains information about the business and reviews"])),
        html.Li(html.Span([html.Code("check-in"), " : contains check-in information for businesses"])),
    ], style={"paddingLeft": "20px"}),

    html.H3("Preview of Datasets", style={"marginTop": "30px"}),

    html.Div([
        html.H4("Review Table:"),
        dash_table.DataTable(
            data=review_preview,
            page_size=5,
            style_table={'overflowX': 'auto', 'minWidth': '100%'},
            style_cell={
                'textAlign': 'left',
                'fontFamily': 'Poppins, sans-serif',
                'fontSize': '14px',
                'minWidth': '120px', 'width': '120px', 'maxWidth': '120px',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
                'whiteSpace': 'nowrap'  # <-- this forces one-line text, no wrapping
            },
            style_header={
                'fontWeight': 'bold',
                'backgroundColor': '#f0f0f0'
            }
        ),
    ], style={"marginBottom": "40px"}),

    html.Div([
        html.H4("User Table:"),
        dash_table.DataTable(
            data=user_preview,
            page_size=5,
            style_table={'overflowX': 'auto', 'minWidth': '100%'},
            style_cell={
                'textAlign': 'left',
                'fontFamily': 'Poppins, sans-serif',
                'fontSize': '14px',
                'minWidth': '120px', 'width': '120px', 'maxWidth': '120px',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
                'whiteSpace': 'nowrap'  # <-- this forces one-line text, no wrapping
            },
            style_header={
                'fontWeight': 'bold',
                'backgroundColor': '#f0f0f0'
            }
        ),
    ], style={"marginBottom": "40px"}),

    html.Div([
        html.H4("Tips Table:"),
        dash_table.DataTable(
            data=tips_preview,
            page_size=5,
            style_table={'overflowX': 'auto', 'minWidth': '100%'},
            style_cell={
                'textAlign': 'left',
                'fontFamily': 'Poppins, sans-serif',
                'fontSize': '14px',
                'minWidth': '120px', 'width': '120px', 'maxWidth': '120px',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
                'whiteSpace': 'nowrap'  # <-- this forces one-line text, no wrapping
            },
            style_header={
                'fontWeight': 'bold',
                'backgroundColor': '#f0f0f0'
            }
        ),
    ], style={"marginBottom": "40px"}),

    html.Div([
        html.H4("Business Table: "),
        dash_table.DataTable(
            data=business_preview,
            page_size=5,
            style_table={'overflowX': 'auto', 'minWidth': '100%'},
            style_cell={
                'textAlign': 'left',
                'fontFamily': 'Poppins, sans-serif',
                'fontSize': '14px',
                'minWidth': '120px', 'width': '120px', 'maxWidth': '120px',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
                'whiteSpace': 'nowrap'  # <-- this forces one-line text, no wrapping
            },
            style_header={
                'fontWeight': 'bold',
                'backgroundColor': '#f0f0f0'
            }
        ),
    ], style={"marginBottom": "40px"}),

    html.Div([
        html.H4("Check-in Table:"),
        dash_table.DataTable(
            data=checkin_preview,
            page_size=5,
            style_table={'overflowX': 'auto', 'minWidth': '100%'},
            style_cell={
                'textAlign': 'left',
                'fontFamily': 'Poppins, sans-serif',
                'fontSize': '14px',
                'minWidth': '120px', 'width': '120px', 'maxWidth': '120px',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
                'whiteSpace': 'nowrap'  # <-- this forces one-line text, no wrapping
            },
            style_header={
                'fontWeight': 'bold',
                'backgroundColor': '#f0f0f0'
            }
        ),
    ], style={"marginBottom": "40px"}),

    # ------- Display Statistics -------
html.H3("Dataset Statistics", style={"marginTop": "40px"}),

# Unique restaurants
html.Div([
    html.H4("Unique Restaurants Count"),
    html.P(f"{unique_restaurants_count:,}")  # Format number with commas
], style={
    "backgroundColor": "#f8f8f8",
    "padding": "15px",
    "borderRadius": "10px",
    "marginBottom": "30px",
    "fontFamily": "Poppins, sans-serif",
    "maxWidth": "300px"
}),

html.Details([
    html.Summary("Show Code"),
    dcc.Markdown('''
    ```python
    unique_restaurants_count = philly_reviews_df['business_id'].nunique()
    ```
    ''', style={"fontFamily": "Poppins, monospace", "whiteSpace": "pre-wrap"})
], style={"marginTop": "10px", "marginBottom": "20px"}),

# Star Ratings Summary
html.Div([
    html.H4("Review Star Ratings Summary"),
    dash_table.DataTable(
        columns=[
            {"name": i, "id": i} for i in stars_summary.columns
        ],
        data=stars_summary.to_dict('records'),
        style_table={'overflowX': 'auto', 'minWidth': '100%'},
        style_cell={
            'textAlign': 'left',
            'fontFamily': 'Poppins, sans-serif',
            'fontSize': '14px',
            'minWidth': '120px', 'width': '150px', 'maxWidth': '150px',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            'whiteSpace': 'nowrap'
        },
        style_header={
            'fontWeight': 'bold',
            'backgroundColor': '#f0f0f0'
        },
        page_size=8
    )
], style={
    "backgroundColor": "#f8f8f8",
    "padding": "20px",
    "borderRadius": "10px",
    "fontFamily": "Poppins, sans-serif",
    "maxWidth": "500px"
}),
html.Details([
    html.Summary("Show Code"),
    dcc.Markdown('''
    ```python
    stars_summary = philly_reviews_df['stars'].describe().round(2).to_frame().reset_index()
    stars_summary.columns = ['Statistic', 'Stars']
    ```
    ''', style={"fontFamily": "Poppins, monospace", "whiteSpace": "pre-wrap"})
], style={"marginTop": "10px", "marginBottom": "20px"}),

html.H3("Distributions", style={"marginTop": "40px"}),

html.Div([
    # First Plot (Stars)
    html.Div([
        dcc.Graph(
            figure=px.histogram(
                philly_reviews_df,
                x="stars",
                nbins=5,
                title="Distribution of Stars",
                labels={"stars": "Star Rating"}
            ).update_layout(bargap=0.3, height=400)  # Set fixed height
        ),
        html.Details([
            html.Summary("Show Code"),
            dcc.Markdown('''
            ```python
            import plotly.express as px
            fig = px.histogram(philly_reviews_df, x='stars', nbins=5)
            fig.update_layout(bargap=0.3)
            fig.show()
            ```
            ''', style={"fontFamily": "Poppins, monospace", "whiteSpace": "pre-wrap"})
        ], style={"marginTop": "10px"})
    ], style={
        "flex": "1",
        "paddingRight": "20px",
        "minWidth": "0"
    }),

    # Second Plot (is_open)
    html.Div([
        dcc.Graph(
            figure=px.histogram(
                philly_reviews_df,
                x="is_open",
                title="Distribution of Restaurant Open Status",
                labels={"is_open": "Is Open (1 = Open, 0 = Closed)"}
            ).update_layout(
                bargap=0.3,
                height=400,
                xaxis=dict(tickmode='array', tickvals=[0, 1])  # Clean up x-ticks
            )
        ),
        html.Details([
            html.Summary("Show Code"),
            dcc.Markdown('''
            ```python
            import plotly.express as px
            fig = px.histogram(philly_reviews_df, x='is_open')
            fig.update_layout(bargap=0.3)
            fig.show()
            ```
            ''', style={"fontFamily": "Poppins, monospace", "whiteSpace": "pre-wrap"})
        ], style={"marginTop": "10px"})
    ], style={
        "flex": "1",
        "paddingLeft": "20px",
        "minWidth": "0"
    }),
], style={
    "display": "flex",
    "flexWrap": "wrap",
    "gap": "20px",
    "marginTop": "20px"
})

    
], style={"fontFamily": "Poppins, sans-serif", "padding": "20px"})
