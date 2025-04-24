import dash
from dash import Dash, html, dcc

external_stylesheets = [
    "https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap"
]

app = Dash(__name__, use_pages=True, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div([
    # Top navbar
    html.Nav([
        html.Div("Yelp Dashboard", style={
            "fontWeight": "600",
            "fontSize": "24px",
            "flex": "1"
        }),
        html.Div([
            dcc.Link("EDA", href="/eda", style={"margin": "0 10px", "textDecoration": "none"}),
            dcc.Link("Analysis", href="/analysis", style={"margin": "0 10px", "textDecoration": "none"}),
            dcc.Link("Visualization", href="/visualization", style={"margin": "0 10px", "textDecoration": "none"}),
        ], style={"display": "flex", "gap": "15px"})
    ], style={
        "display": "flex",
        "justifyContent": "space-between",
        "alignItems": "center",
        "padding": "10px 20px",
        "backgroundColor": "#f8f8f8",
        "borderBottom": "1px solid #ddd",
        "position": "sticky",
        "top": "0",
        "zIndex": "1000",
        "fontFamily": "Poppins, sans-serif"
    }),

    html.Div(dash.page_container, style={"padding": "20px", "fontFamily": "Poppins, sans-serif"})
])


if __name__ == '__main__':
    app.run(debug=True)


