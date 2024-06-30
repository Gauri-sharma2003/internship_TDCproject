from dash import Dash, html, dcc
import dash
import dash_bootstrap_components as dbc
import flask
import logging
import json
from dash.dependencies import Input, Output, State

import csv
from datetime import datetime

#from pages import home, input, analytics
#from pages.Data_Input import is_clicking
#from pages.home import layout, register_callbacks

import decimal
# Set the global precision
decimal.getcontext().prec = 6

import logging

logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(filename)s - %(message)s',
                            filename='./logs.log',
                            filemode='w')



server = flask.Flask(__name__) # define flask app.server
server.secret_key = '123456'  # Set your secret key here
JSW_Group_logo = "./assets/JSW_Group_logo.svg"


app = Dash(__name__, use_pages=True, update_title=None, 
           external_stylesheets=[dbc.themes.BOOTSTRAP,  dbc.icons.FONT_AWESOME], 
           server=server,suppress_callback_exceptions=True)

server = app.server
app.title = 'TDC class prediction model'

logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(filename)s - %(message)s',
                            filename='./logs.log',
                            filemode='w')

dash.register_page("/")



nav = dbc.Nav(
    [
        dbc.NavItem(dbc.NavLink("Home", href="/home", className="page-link")),
    ], fill=False, horizontal='end',
    style={"fontSize": "15px"}
)


footer = html.Footer(
    ["TDC Class Prediction Model - Designed & Developed in collaboration with VJNR PDQC"],
    style={
        'textAlign': 'center',
        'padding': '7.5px',
        'width': '100%',
        'background-color': 'purple',
        'color': 'white',
        'position': 'fixed',  # Set position to fixed
        'left': 0,
        'bottom': 0,
        'marginTop': 'auto',
        'height': '40px',
        'zIndex': 9999  # Ensure footer appears on top of other content
    }
)
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.A(
                html.Div([
                    html.Img(src=JSW_Group_logo, height="60px", style={
                             'margin': "10px 10px 0 0", 'display': 'inline', 'vertical-align': 'bottom'}),
                    html.H1(
                        "TDC Class Prediction Model",
                        style={"margin-bottom": "0px", 'textAlign': 'center', 'color': '#FF1700',
                           'vertical-align': 'super', 'fontSize': '1.5rem', 'display': 'inline'},
                    )
                ], style={'padding': '0'}
                ), href='/home', style={"textDecoration": "none"},
            )
        ], style={'backgroundColor': None}, md=5),
        dbc.Col([
                dbc.Col([
                        nav
                        ], md=7,
                        width={'offset': 5},
                        align='end',
                        class_name='text-end',
                        style={
                    'padding-top': '20px',
                }
                ),
                ])
    ]),
    dbc.Row([
        dbc.Col([
            html.Div([
                dash.page_container,
                footer
            ])
        ])
    ]),
], 
style={'background-image': 'url(./assets/RIST_BG.png)',
          'height': '100vh', 'background-repeat': 'no-repeat',
          'overflow': 'auto',
          'background-size': '100vw 100vh'}, 
fluid=True,
className="p-0",)


if __name__ == '__main__':
    app.run_server(debug=True,port=8080)