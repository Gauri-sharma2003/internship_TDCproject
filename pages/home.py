from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import time
import dash
from src.utils import *

import numpy as np


JSW_Group_logo = "./assets/JSW_Group_logo.svg"

model_config_path = "./config/model_path.json"

dash.register_page(__name__)
log_visit("home")  # Log the visit

chemical_elements = ['C%', 'Mn%', 'S%', 'P%', 'Si%','Al%', 'N%', 'Ti%', 'B%', 'Cr%', 'V%', 'Nb%', 'Mo%',]
pyhical_props = ['YS','UTS','EL','THICKNESS','WIDTH']
applications_list = ['Furnitures and Panels', 'Automotive Internal', 'Export',
       'Automotive Exposed-OEM', 'White Goods', 'General Engineering',
       'Tubes', 'Drum,Bareels,Containers','Other']

model_config = read_json_file(model_config_path)


application_input = dbc.InputGroup([
                            dbc.InputGroupText('Application:', id='app_header',className="fw-bold", style={'backgroundColor':'blue','color':'white'}),
                            dcc.Dropdown(id="application_id", style={'width':400},
                                        options=applications_list, value='')
                        ])

c_per_btn = dbc.InputGroup([
                            dbc.InputGroupText('Carbon %:', id='c_per',className="fw-bold", style={'backgroundColor':'blue','color':'white'}),
                            dbc.Input(id='c_per_id', type='number',value=0.002, required=True,
                                        min=0.0,max=0.351,step=0.01, style={'width':'auto'})
                        ])

mn_per_btn = dbc.InputGroup([
                            dbc.InputGroupText('Manganese %:', id='mn_per',className="fw-bold", style={'backgroundColor':'blue','color':'white'}),
                            dbc.Input(id='mn_per_id', type='number',value=0.17, required=True,
                                        min=0.0,max=4.082,step=0.01, style={'width':'auto'})
                        ])

s_per_btn = dbc.InputGroup([
                            dbc.InputGroupText('Sulfur %:', id='s_per',className="fw-bold", style={'backgroundColor':'blue','color':'white'}),
                            dbc.Input(id='s_per_id', type='number',value=0.006, required=True,
                                        min=0.0,max=0.0273,step=0.01, style={'width':'auto'})
                        ])

p_per_btn = dbc.InputGroup([
                            dbc.InputGroupText('Phosphorus %:', id='p_per',className="fw-bold", style={'backgroundColor':'blue','color':'white'}),
                            dbc.Input(id='p_per_id', type='number',value=0.017, required=True,
                                        min=0.0,max=0.0936,step=0.01, style={'width':'auto'})
                        ])

si_per_btn = dbc.InputGroup([
                            dbc.InputGroupText('Silicon %:', id='si_per',className="fw-bold", style={'backgroundColor':'blue','color':'white'}),
                            dbc.Input(id='si_per_id', type='number',value=0.001, required=True,
                                        min=0.0,max=1.7563,step=0.01, style={'width':'auto'})
                        ])

al_per_btn = dbc.InputGroup([
                            dbc.InputGroupText('Aluminium %:', id='al_per',className="fw-bold", style={'backgroundColor':'blue','color':'white'}),
                            dbc.Input(id='al_per_id', type='number',value=0.045, required=True,
                                        min=0.0,max=0.091,step=0.01, style={'width':'auto'})
                        ])

n_per_btn = dbc.InputGroup([
                            dbc.InputGroupText('Nitrogen %:', id='n_per',className="fw-bold", style={'backgroundColor':'blue','color':'white'}),
                            dbc.Input(id='n_per_id', type='number',value=0.0025, required=True,
                                        min=0.0,max=0.0091,step=0.01, style={'width':'auto'})
                        ])

ti_per_btn = dbc.InputGroup([
                            dbc.InputGroupText('Titanium %:', id='ti_per',className="fw-bold", style={'backgroundColor':'blue','color':'white'}),
                            dbc.Input(id='ti_per_id', type='number',value=0.032, required=True,
                                        min=0.0,max=0.169,step=0.01, style={'width':'auto'})
                        ])

b_per_btn = dbc.InputGroup([
                            dbc.InputGroupText('Boron %:', id='b_per',className="fw-bold", style={'backgroundColor':'blue','color':'white'}),
                            dbc.Input(id='b_per_id', type='number',value=0.001, required=True,
                                        min=0.0,max=0.0039,step=0.01, style={'width':'auto'})
                        ])

cr_per_btn = dbc.InputGroup([
                            dbc.InputGroupText('Chromium %:', id='cr_per',className="fw-bold", style={'backgroundColor':'blue','color':'white'}),
                            dbc.Input(id='cr_per_id', type='number',value=0.018, required=True,
                                        min=0.0,max=0.715,step=0.01, style={'width':'auto'})
                        ])

v_per_btn = dbc.InputGroup([
                            dbc.InputGroupText('Vanadium %:', id='v_per',className="fw-bold", style={'backgroundColor':'blue','color':'white'}),
                            dbc.Input(id='v_per_id', type='number',value=0.001, required=True,
                                        min=0.0,max=0.1664,step=0.01, style={'width':'auto'})
                        ])

nb_per_btn = dbc.InputGroup([
                            dbc.InputGroupText('Niobium %:', id='nb_per',className="fw-bold", style={'backgroundColor':'blue','color':'white'}),
                            dbc.Input(id='nb_per_id', type='number',value=0.006, required=True,
                                        min=0.0,max=0.1079,step=0.01, style={'width':'auto'})
                        ])

mo_per_btn = dbc.InputGroup([
                            dbc.InputGroupText('Molybdenum %:', id='mo_per',className="fw-bold", style={'backgroundColor':'blue','color':'white'}),
                            dbc.Input(id='mo_per_id', type='number',value=0.001, required=True,
                                        min=0.0,max=0.2275,step=0.01, style={'width':'auto'})
                        ])

thikness_btn = dbc.InputGroup([
                            dbc.InputGroupText('Thickness :', id='thk',className="fw-bold", style={'backgroundColor':'blue','color':'white'}),
                            dbc.Input(id='thk_id', type='number',value=2.25, required=True,
                                        min=0.245,max=3.0,step=0.01, style={'width':'auto'})
                        ])

width_btn = dbc.InputGroup([
                            dbc.InputGroupText('Width :', id='width',className="fw-bold", style={'backgroundColor':'blue','color':'white'}),
                            dbc.Input(id='width_id', type='number',value=1606, required=True,
                                        min=626,max=2461,step=0.01, style={'width':'auto'})
                        ])

ys_btn = dbc.InputGroup([
                            dbc.InputGroupText('Yield Strength :', id='ys',className="fw-bold", style={'backgroundColor':'blue','color':'white'}),
                            dbc.Input(id='ys_id', type='number',value=119.133, required=True,
                                        min=25,max=1089,step=0.01, style={'width':'auto'})
                        ])

uts_btn = dbc.InputGroup([
                            dbc.InputGroupText('Ultimate Tensile Strength :', id='uts',className="fw-bold", style={'backgroundColor':'blue','color':'white'}),
                            dbc.Input(id='uts_id', type='number',value=119.133, required=True,
                                        min=83,max=1421,step=0.01, style={'width':'auto'})
                        ])

el_btn = dbc.InputGroup([
                            dbc.InputGroupText('Elongation :', id='el',className="fw-bold", style={'backgroundColor':'blue','color':'white'}),
                            dbc.Input(id='el_id', type='number',value=2.975, required=True,
                                        min=-1.75,max=91.31,step=0.01, style={'width':'auto'})
                        ])

btn_prediction = dbc.Button(
    children=[
        html.Span([
            html.I(className="fa-solid fa-predictive-analysis fa-3x", style={"display": "block"}), 
            "Predict"
        ])
    ],
    outline=True,
    id='btn_prediction',
    color='primary',
    href='/home',
    style={'fontSize': 25, 'textAlign': 'center', 'height': '100%', "width": "100%"}
)

prediction_output = html.Div(id='prediction_output')

application_container = dbc.Container([
    dbc.Row([
        html.Div(style={'borderTop': '2px solid black', 'height': '2px', 'margin': '10px'}),
    ]),
    dbc.Row([
        dbc.Col(application_input, width={"size": 6, "offset": 3}, className="text-center")
    ], justify="center", align="center")
], fluid=True, style={'padding': '0 150px'})

all_chemistry_container = dbc.Container([
        dbc.Row([
            html.Div(style={'borderTop': '2px solid black', 'height': '2px', 'margin': '10px'}),
            ]),
        html.H4('Required Chemical Properties', style={'textAlign': 'center','fontSize': 30,'color':'red'}),
        dbc.Row([
            html.Div(style={'borderTop': '2px solid black', 'height': '2px', 'margin': '10px'}),
            ]),
        dbc.Row([
            dbc.Col([
                c_per_btn
            ], width='auto'),
            dbc.Col([
                mn_per_btn
            ], width='auto'),
            dbc.Col([
                s_per_btn
            ], width='auto'),
            dbc.Col([
                p_per_btn
            ], width='auto'),
        ],className="align-items-center justify-content-center"),
        html.Br(),
        dbc.Row([
            dbc.Col([
                si_per_btn
            ], width='auto'),
            dbc.Col([
                al_per_btn
            ], width='auto'),
            dbc.Col([
                n_per_btn
            ], width='auto'),
            dbc.Col([
                ti_per_btn
            ], width='auto'),
            dbc.Col([
                b_per_btn
            ], width='auto'),
        ],className="align-items-center justify-content-center"),
        html.Br(),
        dbc.Row([html.Div()]),
        dbc.Row([
            dbc.Col([
                cr_per_btn
            ], width='auto'),
            dbc.Col([
                v_per_btn
            ], width='auto'),
            dbc.Col([
                nb_per_btn
            ], width='auto'),
            dbc.Col([
                mo_per_btn
            ], width='auto')
        ],className="align-items-center justify-content-center"),
        dbc.Row([
            html.Div(style={'borderTop': '2px solid black', 'height': '2px', 'margin': '10px'}),
            ]),
    ], fluid=True, style={'padding': '0 150px'})

all_physical_container = dbc.Container([
        dbc.Row([
            html.Div(style={'borderTop': '2px solid black', 'height': '2px', 'margin': '10px'}),
            ]),
        html.H4('Required Physical Properties', style={'textAlign': 'center','fontSize': 30,'color':'red'}),
        dbc.Row([
            html.Div(style={'borderTop': '2px solid black', 'height': '2px', 'margin': '10px'}),
            ]),
        dbc.Row([
            dbc.Col([
                thikness_btn
            ], width='auto'),
            dbc.Col([
                width_btn
            ], width='auto'),
        ],className="align-items-center justify-content-center"),
        html.Br(),
        dbc.Row([
            dbc.Col([
                ys_btn
            ], width='auto'),
            dbc.Col([
                uts_btn
            ], width='auto'),
            dbc.Col([
                el_btn
            ], width='auto'),
        ],className="align-items-center justify-content-center"),
        dbc.Row([
            html.Div(style={'borderTop': '2px solid black', 'height': '2px', 'margin': '10px'}),
            ]),
    ], fluid=True, style={'padding': '0 150px'})


btn_prediction_container = dbc.Container([
    dbc.Row(
        dbc.Col(
            btn_prediction
        ),
        justify="center",
        align="center",  # Full viewport height
    ),
], fluid=True, 
style={
    'padding': '0 150px',
    'display': 'flex',  # Enable Flexbox layout
    'justify-content': 'center',  # Center horizontally
})



output_container = dbc.Container([
    dbc.Row(
        dbc.Col(
            prediction_output
        ),
        justify="center",
        align="center",  # Full viewport height
    ),
], fluid=True, 
style={
    'padding': '0 150px',
    'display': 'flex',  # Enable Flexbox layout
    'justify-content': 'center',  # Center horizontally
})

layout = dbc.Container([
    html.Hr(),
    dbc.Row([
            application_container
    ]),
    dbc.Row([
            all_chemistry_container
    ]),
    dbc.Row([
            all_physical_container
    ]),
    dbc.Row([
            btn_prediction_container
    ]),
    dbc.Row([
            output_container
    ])
], fluid=True, style={'padding': '0 150px'})



@callback(
    Output('prediction_output', 'children'),
    Input('btn_prediction', 'n_clicks'),
    State('application_id', 'value'),
    State('c_per_id', 'value'),
    State('mn_per_id', 'value'),
    State('s_per_id', 'value'),
    State('p_per_id', 'value'),
    State('si_per_id', 'value'),
    State('al_per_id', 'value'),
    State('n_per_id', 'value'),
    State('ti_per_id', 'value'),
    State('b_per_id', 'value'),
    State('cr_per_id', 'value'),
    State('v_per_id', 'value'),
    State('nb_per_id', 'value'),
    State('mo_per_id', 'value'),
    State('thk_id', 'value'),
    State('width_id', 'value'),
    State('ys_id', 'value'),
    State('uts_id', 'value'),
    State('el_id', 'value'),
)
def make_prediction(n_clicks, app, c_per, mn_per, s_per, p_per, si_per, al_per, n_per, ti_per, b_per, cr_per, v_per, nb_per, mo_per, thk, width, ys, uts, el):
    if n_clicks:
        # Prepare input features
        print("app : ",app)
        
        input_features = np.array([[thk, width, ys, uts, el,c_per, mn_per, s_per, p_per, si_per, al_per, n_per, ti_per, b_per, cr_per, v_per, nb_per, mo_per]])
        
        # Scale the input features
        
        with open("models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
            
        scaled_features = scaler.transform(input_features)
        
        model_path = model_config[app]
        # Load your model
        model = load_model_from_pickle(model_path)
        
        # Make prediction
        prediction = model.predict(scaled_features)
        #prediction = ["TDC-AU"]
        
        # Get probability estimates
        confidence = model.predict_proba(input_features)
        confidence_percentage = np.max(confidence) * 100  # Get the maximum probability
        
        # Convert prediction to a readable format
        output_class = f"{prediction[0]}"
        confidence_value = f"Confidence: {confidence_percentage:.2f}%"
        
        return html.Div([
    html.Hr(),
    html.Div(style={'borderTop': '2px solid black', 'height': '2px', 'margin': '10px'}),
    html.H3("Predicted Class is", style={'color': 'red', 'textAlign': 'center'}),  # Align to center
    html.H4(output_class, style={'textAlign': 'center','color': 'blue'}),  # Align to center
    html.P(confidence_value, style={'textAlign': 'center'}),  # Align to center
    html.Div(style={'borderTop': '2px solid black', 'height': '2px', 'margin': '10px'}),
], style={'textAlign': 'center'})  # Align the entire Div content to center

    return html.Div()