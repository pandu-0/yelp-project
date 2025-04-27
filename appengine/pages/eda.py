import dash
from dash import html, dcc, dash_table
import pandas as pd
from google.cloud import storage
import os
from io import StringIO

dash.register_page(__name__, path='/eda', name='EDA')

# --------- Load JSON files ---------
def get_csv_from_gcs(bucket_name, source_blob_name):
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
review_preview = review_df.head().to_dict('records')
user_preview = user_df.head().to_dict('records')
tips_preview = tips_df.head().to_dict('records')
business_preview = business_df.head().to_dict('records')
checkin_preview = checkin_df.head().to_dict('records')

# --------- Layout ---------
layout = html.Div([
    html.H1("EDA", style={"fontWeight": "600"}),

    html.H3("Yelp Dataset Summary:", style={"marginTop": "30px"}),
    html.P(html.Span(["The Yelp Open dataset comes with 5 files in ", html.Code("json"), " format:", ])),
    html.Ul([
        html.Li(html.Span([html.Code("Review"), " : contains reviews and related data left by a customer for a business"])),
        html.Li(html.Span([html.Code("User"), " : contains the users and their profile information"])),
        html.Li(html.Span([html.Code("Tips"), " : tip or comment left by a user for a business"])),
        html.Li(html.Span([html.Code("Business"), " : contains information about the business and reviews"])),
        html.Li(html.Span([html.Code("check-in"), " : contains check-in information for businesses"])),
    ], style={"paddingLeft": "20px"}),

    html.H3("Preview of Datasets", style={"marginTop": "30px"}),

    html.Div([
        html.H4("Review Dataset:"),
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
        html.H4("User Dataset:"),
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
        html.H4("Tips Dataset:"),
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
        html.H4("Business Dataset: "),
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
        html.H4("Check-in Dataset:"),
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
    ], style={"marginBottom": "40px"})
    
], style={"fontFamily": "Poppins, sans-serif", "padding": "20px"})
