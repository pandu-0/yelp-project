from dash import html, dcc
import dash

dash.register_page(__name__, path='/analysis', name='Analysis')

layout = html.Div([
    html.H2("Analysis"),
    html.P("Your analysis components go here."),
    html.Br(),
    dcc.Link(html.Button("Back to Home"), href="/")
])

