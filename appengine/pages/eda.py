import dash
from dash import dcc, html, dash_table, Output, Input, callback
import pandas as pd
import plotly.express as px
from utils import get_json_from_gcs, GCLOUD_BUCKET

dash.register_page(__name__, path='/eda', name='EDA')

# --------- Load JSON files ---------
review_df = pd.read_json('https://raw.githubusercontent.com/pandu-0/yelp-project/refs/heads/main/assets/review_head.json',
                        lines=True)
user_df = pd.read_json('https://raw.githubusercontent.com/pandu-0/yelp-project/refs/heads/main/assets/user_head.json',
                        lines=True)
tips_df = pd.read_json('https://raw.githubusercontent.com/pandu-0/yelp-project/refs/heads/main/assets/tips_head.json',
                        lines=True)

business_df = pd.read_json('https://raw.githubusercontent.com/pandu-0/yelp-project/refs/heads/main/assets/business_head.json',
                        lines=True)
# Convert nested columns to strings (if don't do this, it causes issues with displaying in DataTable)
business_df['attributes'] = business_df['attributes'].astype(str)
business_df['hours'] = business_df['hours'].astype(str)

checkin_df = pd.read_json('https://raw.githubusercontent.com/pandu-0/yelp-project/refs/heads/main/assets/checkin_head.json',
                        lines=True)

# show a few rows (head) for preview
review_preview = review_df.to_dict('records')
user_preview = user_df.to_dict('records')
tips_preview = tips_df.to_dict('records')
business_preview = business_df.to_dict('records')
checkin_preview = checkin_df.to_dict('records')

# --------- Get statistics ---------
unique_restaurants_count = 5852
stars_summary = get_json_from_gcs(GCLOUD_BUCKET, "eda/stars_describe.json")
stars_summary.columns = ['statistic', 'stars']  # Rename columns for nice table display

stars_count = get_json_from_gcs(GCLOUD_BUCKET, "eda/stars_count.json")
is_open_count = get_json_from_gcs(GCLOUD_BUCKET, "eda/is_open_count.json")

restaurant_mean_stars = get_json_from_gcs(GCLOUD_BUCKET, "eda/restaurant_mean_stars.json")
restaurant_review_count = get_json_from_gcs(GCLOUD_BUCKET, "eda/restaurant_review_count.json")

year_and_is_open_count = get_json_from_gcs(GCLOUD_BUCKET, "eda/year_and_is_open.json")

philly_restaurants = get_json_from_gcs(GCLOUD_BUCKET, "eda/philly_restaurants.json")

# Convert is_open to string for clarity
philly_restaurants['is_open'] = philly_restaurants['is_open'].map({1: "Open", 0: "Closed"})

@callback(
    Output("map-graph", "figure"),
    Input("color-radio", "value")
)
def update_map(color_by):
    df = philly_restaurants.copy()

    # Drop null coordinates
    df = df[df["latitude"].notnull() & df["longitude"].notnull()]

    if color_by == "review_count":
        bins = [0, 10, 50, 100, 250, 500, 1000, 2000, float("inf")]
        labels = ["0-10", "10-50", "50-100", "100-250", "250-500", "500-1K", "1K-2K", "2K+"]

        # Step 1: Cut and assign ordered category
        df["review_bucket"] = pd.cut(
            df["review_count"],
            bins=bins,
            labels=labels,
            include_lowest=True
        )

        df["review_bucket"] = pd.Categorical(df["review_bucket"], categories=labels, ordered=False)

        color_col = "review_bucket"

        color_sequence = [
            "#8b2c8e", "#68217a", "#2b0f54", "#471365", 
            "#d85cb0", "#b03aa3", "#f89ac3", "#ffd1da"
        ]
    else:
        color_col = color_by
        color_sequence = None  # Let Plotly handle default

    fig = px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        color=color_col,
        hover_name="name",
        hover_data={"categories": True, "stars": True, "review_count": True},
        zoom=10,
        center={"lat": 39.9526, "lon": -75.1652},
        height=600,
        color_discrete_sequence=color_sequence
    )

    fig.update_layout(
        mapbox_style="carto-positron",
        margin={"r":0, "t":40, "l":0, "b":0},
        title=f"Philadelphia Restaurants Colored by {color_by.replace('_', ' ').title()}",
        title_x=0.5
    )

    return fig

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
        html.H4([
            "Unique Restaurants Count: ",
            html.Span(f"{unique_restaurants_count:,}", style={"fontWeight": "normal"})
        ])
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

    html.Div([
        # Left: Star Ratings Summary
        html.Div([
            html.H4("Review Star Ratings Summary"),
            dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in stars_summary.columns],
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
            ),
            html.Details([
                html.Summary("Show Code"),
                dcc.Markdown('''
                ```python
                stars_summary = philly_reviews_df['stars'].describe().round(2).to_frame().reset_index()
                stars_summary.columns = ['Statistic', 'Stars']
                ```
                ''', style={"fontFamily": "Poppins, monospace", "whiteSpace": "pre-wrap"})
            ]),
            html.Details([
            html.Summary("Description"),
            html.P("The table shows statistics of the Review Stars column. "
                   "As per the table, we can see that most of the reviews are between 3 and 5 stars.")
            ])
        ], style={
            "backgroundColor": "#f8f8f8",
            "padding": "20px",
            "borderRadius": "10px",
            "fontFamily": "Poppins, sans-serif",
            "flex": "1",
            "minWidth": "0"
        }),

        # Right: Violin Plot
        html.Div([
            dcc.Graph(figure=px.violin(
            restaurant_mean_stars,
            y='stars',
            box=True,
            title='Distribution of Mean Restaurant Review Stars',
            labels={'stars': 'Star Rating'}
            ).update_layout(height=400)),
            html.Details([
            html.Summary("Show Code"),
            dcc.Markdown('''
            ```python
            restaurant_mean_stars = philly_reviews_df.groupby('business_id')['stars'].mean().to_frame().reset_index()
            fig = px.violin(restaurant_mean_stars, y='stars', box=True)
            fig.update_layout(height=400)
            fig.show()
            ```
            ''', style={"fontFamily": "Poppins, monospace", "whiteSpace": "pre-wrap"})
            ]),
            html.Details([
            html.Summary("Description"),
            html.P("This violin plot shows the distribution of the mean star ratings for restaurants. "
                   "As per the graph, most restaurants have a mean rating between 3.5 and 4.5 stars. ")
            ])
        ], style={
            "flex": "1",
            "minWidth": "0",
            "display": "flex",
            "flexDirection": "column",
            "justifyContent": "space-between"
        }),
        ], style={
        "display": "flex",
        "gap": "20px",
        "alignItems": "stretch",  # Ensures both children are same height
        "marginTop": "20px"
    }),

    html.H3("Distributions", style={"marginTop": "40px"}),

    html.Div([
        # First Plot (Stars)
        html.Div([
            dcc.Graph(figure=px.bar(
            stars_count,
            x='stars',
            y='count',
            title='Countplot of Restaurant Review Stars',
            labels={'stars': 'Star Rating', 'count': 'Review Count'}
            ).update_layout(
            height=400,
            bargap=0.3
            )),
            html.Details([
            html.Summary("Show Code"),
            dcc.Markdown('''
            ```python
            stars_count = philly_reviews_df['stars'].value_counts().reset_index()
            stars_count.columns = ['stars', 'count']
            stars_count = stars_count.sort_values(by='stars')
            fig = px.bar(stars_count, x='stars', y='count')
            fig.update_layout(bargap=0.3)
            fig.show()
            ```
            ''', style={"fontFamily": "Poppins, monospace", "whiteSpace": "pre-wrap"})
            ], style={"marginTop": "10px"}),
            html.Details([
            html.Summary("Description"),
            html.P("The graph depicts the distribution of the Review Stars. "
                   "As per the graph, a 5 star rating was the most popular rating given.")
            ], style={"marginTop": "10px"})
        ], style={
            "flex": "1",
            "paddingRight": "20px",
            "minWidth": "0"
        }),

        # Second Plot (is_open)
        html.Div([
            dcc.Graph(
                figure=px.bar(
                    is_open_count,
                    x="is_open",
                    y="count",
                    title="Countplot of Restaurant Review Open Status",
                    labels={"is_open": "Is Open (1 = Open, 0 = Closed)"}
                ).update_layout(
                    height=400,
                    xaxis=dict(tickmode='array', tickvals=[0, 1]),  # Clean up x-ticks
                    bargap=0.3
                )
            ),
            html.Details([
                html.Summary("Show Code"),
                dcc.Markdown('''
                ```python
                import plotly.express as px
                is_open_count = philly_restaurant_reviews['is_open'].value_counts().to_frame().reset_index()
                fig = px.bar(is_open_count, x='is_open', y='count')
                fig.update_layout(bargap=0.3)
                fig.show()
                ```
                ''', style={"fontFamily": "Poppins, monospace", "whiteSpace": "pre-wrap"})
            ], style={"marginTop": "10px"}),
            html.Details([
                html.Summary("Description"),
                html.P("The above graph shows the number of reviews among open and close resturants where 0 is closed and 1 is open. "
                       "As per the graph, most of the reviews are from open restaurants."
                       )
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
    }),
    
    html.Div([
        # Violin plot of review count grouped by buisiness_id
        html.Div([
            dcc.Graph(
                figure=px.violin(
                    restaurant_review_count,
                    y='review',
                    box=True,
                    title="Distribution of Number of Reviews per Restaurant",
                    labels={'review': 'Number of Reviews'}
                ).update_layout(
                    height=400,
                )
            ),
            # show the plot code
            html.Details([
                html.Summary("Show Code"),
                dcc.Markdown('''
                ```python
                restaurant_review_count = philly_reviews_df.groupby('business_id')['review'].count().to_frame().reset_index()
                fig = px.violin(restaurant_review_count, y='review', box=True),
                fig.update_layout(height=400),
                fig.show()
                ```
                ''', style={"fontFamily": "Poppins, monospace", "whiteSpace": "pre-wrap"})
            ], style={"marginTop": "10px"}),
            html.Details([
                html.Summary("Description"),
                html.P("the above graph depicts the distribution of the number of reviews for each restaurant. "
                       "As per the graph, most restaurants have between 0 and 40 reviews, showing that many resturants do not have a lot of reviews. ")
            ], style={"marginTop": "10px"})
        ], style={
            "flex": "1",
            "paddingLeft": "20px",
            "minWidth": "0"
        }),

        # Countplot of review count grouped by year and hue to is_open
        html.Div([
            dcc.Graph(
                figure=px.histogram(
                    year_and_is_open_count,
                    x='year',
                    color='is_open',
                    barmode='group',
                    category_orders={'year': sorted(year_and_is_open_count['year'].unique())},
                    labels={'is_open': 'Is Open', 'year': 'Year', 'count': 'Count'},
                    title='Restaurant Reviews by Year and Open Status'
                ).update_layout(
                    xaxis_title='Year',
                    yaxis_title='Count',
                    xaxis_tickangle=45,
                    bargap=0.3,
                    height=400
                )
            ),
            # show the plot code
            html.Details([
                html.Summary("Show Code"),
                dcc.Markdown('''
                    ```python
                    plt.figure(figsize=(12, 6))  # Make the plot wider
                    sns.countplot(philly_restaurant_reviews, x='year', hue='is_open')
                    plt.xticks(rotation=45)      # Rotate x-axis labels 45 degrees
                    plt.tight_layout()           # Adjust layout to prevent label cutoff
                    plt.show()         
                    ```
                ''', style={"fontFamily": "Poppins, monospace", "whiteSpace": "pre-wrap"})
            ], style={'marginTop': '10px'}),
            html.Details([
                html.Summary("Description"),
                html.P("The above graph shows the number of reviews by year and open status. "
                       "As per the graph, most of the reviews are from open restaurants in 2014 to 2019. ")
            ], style={"marginTop": "10px"})
        ],  style={
            "flex": "1",
            "paddingLeft": "20px",
            "minWidth": "0"
        })
    ], style={
        "display": "flex",
        "flexWrap": "wrap",
        "gap": "20px",
        "marginTop": "20px"
    }),

    # ------- Map -------
    html.H3("Map of Philadelphia Restaurants", style={"marginTop": "40px"}),
    html.Div([
        html.Label("Select Coloring Option:", style={"fontWeight": "bold", "fontSize": "16px"}),
        dcc.RadioItems(
            id="color-radio",
            options=[
                {"label": "Open Status", "value": "is_open"},
                {"label": "Star Rating", "value": "stars"},
                {"label": "Review Count", "value": "review_count"},
            ],
            value="is_open",
            labelStyle={"display": "inline-block", "margin-right": "15px"},
            inputStyle={"margin-right": "6px"},
            style={"margin-bottom": "10px"},
        ),
        dcc.Graph(id="map-graph"),
        html.P("The map shows the resturants in Philadelphia by open status, star rating, and review count. "
               "By hovering over the dots, more infomation, such as the name of the resturant and category, can be seen. "
               , 
               style={"marginTop": "10px", "fontSize": "17px"})
    ])
    
], style={"fontFamily": "Poppins, sans-serif", "padding": "20px"})
