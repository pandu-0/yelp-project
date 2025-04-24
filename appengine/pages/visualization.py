from dash import html, dcc
import dash

dash.register_page(__name__, path='/visualization', name='Visualization')

layout = html.Div([
    html.H2("Visualization"),
    html.P("Your visualizations go here."),
    html.Br(),
    dcc.Link(html.Button("Back to Home"), href="/")
])

