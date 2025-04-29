from dash import html, dcc
from dash.dependencies import Input, Output
import dash
import pandas as pd
import plotly.express as px

dash.register_page(__name__, path='/visualization', name='Visualization')
df = pd.read_csv('preview_datasets/Adjective_correlation_rating')

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


# App layout
layout = html.Div([
    html.H2("Visualization"),

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

    html.Br(),
    dcc.Link(html.Button("Back to Home"), href="/")
])


