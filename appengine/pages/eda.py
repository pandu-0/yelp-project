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
review_df = pd.read_json('path/to/review.json', lines=True)
user_df = pd.read_json('path/to/user.json', lines=True)
tips_df = pd.read_json('path/to/tip.json', lines=True)
business_df = pd.read_json('path/to/business.json', lines=True)
checkin_df = pd.read_json('path/to/checkin.json', lines=True)

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
    html.P("The Yelp Open dataset comes with 5 files in `json` format:"),
    html.Ul([
        html.Li(html.Span([html.Code("Review"), " : Contains reviews and related data left by a customer for a business"])),
        html.Li(html.Span([html.Code("User"), " : Contains the users and their profile information"])),
        html.Li(html.Span([html.Code("Tips"), " : Tip or comment left by a user for a business"])),
        html.Li(html.Span([html.Code("Business"), " : Contains information about the business and reviews"])),
        html.Li(html.Span([html.Code("check-in"), " : Contains check-in information for businesses"])),
    ], style={"paddingLeft": "20px"}),

    html.H3("Preview of Datasets", style={"marginTop": "30px"}),

    html.Div([
        html.H4("Review Dataset Preview"),
        dash_table.DataTable(
            data=review_preview,
            page_size=5,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'fontFamily': 'Poppins, sans-serif', 'fontSize': '14px'}
        ),
    ], style={"marginBottom": "40px"}),

    html.Div([
        html.H4("User Dataset Preview"),
        dash_table.DataTable(
            data=user_preview,
            page_size=5,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'fontFamily': 'Poppins, sans-serif', 'fontSize': '14px'}
        ),
    ], style={"marginBottom": "40px"}),

    html.Div([
        html.H4("Tips Dataset Preview"),
        dash_table.DataTable(
            data=tips_preview,
            page_size=5,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'fontFamily': 'Poppins, sans-serif', 'fontSize': '14px'}
        ),
    ], style={"marginBottom": "40px"}),

    html.Div([
        html.H4("Business Dataset Preview"),
        dash_table.DataTable(
            data=business_preview,
            page_size=5,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'fontFamily': 'Poppins, sans-serif', 'fontSize': '14px'}
        ),
    ], style={"marginBottom": "40px"}),

    html.Div([
        html.H4("Check-in Dataset Preview"),
        dash_table.DataTable(
            data=checkin_preview,
            page_size=5,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'fontFamily': 'Poppins, sans-serif', 'fontSize': '14px'}
        ),
    ], style={"marginBottom": "40px"})
    
], style={"fontFamily": "Poppins, sans-serif", "padding": "20px"})
