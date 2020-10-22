import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
import pickle

col_alias = {'ProductCD':'Product CD',
             'card4':'Card Network Processor',
             'card6':'Card Type',
             'DeviceType':'Device Type',
             'DeviceName':'Device Name',
             'sys_id_30':'System',
             'browser_id_31':'Browser',
             'screen_id-33':'Screen Size',
             'P_emaildomain_bin':'Primary Email Domain BIN',
             'P_emaildomain_suffix':'Primary Email Domain Suffix',
             'R_emaildomain_bin':'Alternative Email Domain BIN',
             'R_emaildomain_suffix':'Alternative Email Domain Suffix',
             'Transaction_Weekdays':'Transaction Weekday',
             'Transaction_Hours':'Transaction Hour of Day',
             'Transaction_Days':'Transaction Day of Month',
             'Transaction_Months':'Transaction Month'
}

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# the style arguments for the sidebar.
SIDEBAR_STYLE = {
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 1,
    'width': '20%',
    'padding': '20px 10px',
    'background-color': '#f8f9fa',
    'overflow-y': 'scroll',
}

# the style arguments for the main content page.
CONTENT_STYLE = {
    'margin-left': '25%',
    'margin-right': '5%',
    'top': 0,
    'padding': '20px 10px'
}

TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#191970'
}

CARD_TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#0074D9'
}


# load data
filename = os.path.join(os.getcwd(), '../output/cleaned_train.pkl')
print(filename)
train = pickle.load(open(filename, "rb"))

# load logo
BANK_LOGO = os.path.join('bank-logo.png')

controls = dbc.FormGroup(
    [
        html.P('Card Type', style={
            'textAlign': 'center'
        }),
        dcc.RadioItems(
            id = 'card_type',
            options=[
                {'label': 'Debit', 'value': 'debit'},
                {'label': 'Credit', 'value': 'credit'},
                {'label': 'Others', 'value': 'None'}
            ],
            value='debit'
        ),
        html.Br(),
        
        html.P('Card Network Processor', style={
            'textAlign': 'center'
        }),
        dcc.Dropdown(
            id='card_processor',
            options=[{'label': x, 'value':x} for x in train['card4'].unique()],
            value=['visa'],  # default value
            multi=True
        ),
        dcc.Checklist(id='select-all-cardprocessors', options=[{'label': 'Select All', 'value': 1}], value=[]),
        html.Br(),

        html.P('Product CD', style={
            'textAlign': 'center'
        }),
        dcc.Dropdown(
            id='product_cd',
            options=[{'label': x, 'value':x} for x in train['ProductCD'].unique()],
            value=['W'],  # default value
            multi=True
        ),
        dcc.Checklist(id='select-all-products', options=[{'label': 'Select All', 'value': 1}], value=[]),
        html.Br(),

        dbc.Button(
            id='submit_button',
            n_clicks=0,
            children='Submit',
            color='primary',
            block=True
        ),
    ]
)

# sidebar
sidebar = html.Div(
    [
        html.H2('Card Filters', style=TEXT_STYLE),
        html.Hr(),
        controls
    ],
    style=SIDEBAR_STYLE,
)


content_first_row = dbc.Row([
    dbc.Col(
        dbc.Card(
            [

                dbc.CardBody(
                    [
                        html.H4(id='card_title_1', children=['Card Title 1'], className='card-title',
                        style=CARD_TEXT_STYLE),
                        html.P(id='card_text_1', children=['# transactions'], style=CARD_TEXT_STYLE),
                    ]
                )
            ]
        ),
        md=4
    ),
    dbc.Col(
        dbc.Card(
            [

                dbc.CardBody(
                    [
                        html.H4(id='card_title_2', children=['Card Title 2'], className='card-title', 
                        style=CARD_TEXT_STYLE),
                        html.P(id='card_text_2', children=['% Fraud'], style=CARD_TEXT_STYLE),
                    ]
                ),
            ]

        ),
        md=4
    ),
    dbc.Col(
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H4(id='card_title_3', children=['Card Title 3'], className='card-title', 
                        style=CARD_TEXT_STYLE),
                        html.P(id='card_text_3', children=['Average Transaction Amount'], style=CARD_TEXT_STYLE),

                    ]
                ),
            ]

        ),
        md=4
    )
])


content_second_row = dbc.Row(
    [
        dbc.Col(
            dcc.Graph(id='device_type_bar'), md=6
        ),
        dbc.Col(
            dcc.Graph(id='device_name_bar'), md=6
        )
    ]
)


content_third_row = dbc.Row(
    [
        dbc.Col(
            dbc.Tabs(
            [
                dbc.Tab(label="Hour of Day", tab_id="hour"),
                dbc.Tab(label="Weekday", tab_id="weekday"),
                dbc.Tab(label="Day of Month", tab_id="day"),
                dbc.Tab(label="Month", tab_id="month"),
            ],
            id="tabs",
            active_tab="hour",
            ),
            md=12
        ),
        dbc.Col(
            html.Div(id="tab-content", className="p-4"),
            md=12
        ),

    ]
)

content = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(html.Img(src=app.get_asset_url(BANK_LOGO), style = {'height':'120px'}), md=2),
                dbc.Col(html.H2('Fraud Detection Demographic Dashboard', style=TEXT_STYLE), md=8),
                dbc.Col(md=2),
            ],
            align="center",
            no_gutters=True,
            ),
        html.Hr(),
        content_first_row,
        content_second_row,
        html.Hr(),
        dbc.Row([
            html.H4('Fraud Ratio by Time')
        ],
        style = {'margin-top':'20px', 'margin-bottom':'50px'}),
        content_third_row,
        # content_fourth_row
    ],
    style=CONTENT_STYLE
)

app.layout = html.Div([sidebar, content])

@app.callback(
    Output('card_processor', 'value'),
    [Input('select-all-cardprocessors', 'value')],
    [State('card_processor', 'options'),
     State('card_processor', 'value')])
def select_all_countries(selected, options, values):
    if len(selected) == 1:
        return [i['value'] for i in options]
    elif len(values) != 0:
        return values
    else:
        return ['visa']

@app.callback(
    Output('product_cd', 'value'),
    [Input('select-all-products', 'value')],
    [State('product_cd', 'options'),
     State('product_cd', 'value')])
def select_all_systemhardwares(selected, options, values):
    if len(selected) == 1:
        return [i['value'] for i in options]
    elif len(values) != 0:
        return values
    else:
        return ['W']


@app.callback(
    [Output('card_title_1', 'children'),
    Output('card_title_2', 'children'),
    Output('card_title_3', 'children'),
    Output('device_type_bar', 'figure'),
    Output('device_name_bar', 'figure')],
    [Input('submit_button', 'n_clicks')],
    [State('card_type', 'value'), 
    State('card_processor', 'value'), 
    State('product_cd', 'value')
     ])
def update_filters(n_clicks, card_type, card_processor, product_cd):
    filtered_train = train[(train['ProductCD'].isin(product_cd))&
                            (train['card6'] == card_type)&
                            (train['card4']).isin(card_processor)]

    fraud_ratio = str(np.round(filtered_train.isFraud.mean() * 100, 2))+'%'
    avg_transaction = int(np.round(train.TransactionAmt.mean() * 100))

    # device type plot
    fig1 = plot_dist_and_fraud_pct_plotly(train, 'DeviceType', top_n = 5)

    # device name plot
    fig2 = plot_dist_and_fraud_pct_plotly(train, 'DeviceName', top_n = 5)

    return len(filtered_train), fraud_ratio, avg_transaction, fig1, fig2

@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "active_tab")],
)
def render_tab_content(active_tab):
    if active_tab == 'hour':
        fig = plot_time_dist_and_fraud_pct_plotly(train, 'Transaction_Hours')
        return dcc.Graph(figure=fig)
    elif active_tab == 'weekday':
        fig = plot_time_dist_and_fraud_pct_plotly(train, 'Transaction_Weekdays')
        return dcc.Graph(figure=fig)
    elif active_tab == 'day':
        fig = plot_time_dist_and_fraud_pct_plotly(train, 'Transaction_Days')
        return dcc.Graph(figure=fig)
    elif active_tab == 'month':
        fig = plot_time_dist_and_fraud_pct_plotly(train, 'Transaction_Months')
        return dcc.Graph(figure=fig)
    
    return "No tab selected"


# helper functions
def plot_dist_and_fraud_pct_plotly(df, col, top_n = 5):
    df_plot = df.groupby(col).agg(records_num = ('isFraud', 'size'),
                                     records_ratio = ('isFraud',lambda x: np.round(len(x)/len(df)*100, 2)), 
                                     fraud_ratio = ('isFraud', lambda x: np.round(x.mean()*100, 2))).reset_index()
    df_plot = df_plot.sort_values('records_num', ascending=False).head(top_n)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=df_plot[col],
            y=df_plot.records_num,
            text=[str(x)+'%' for x in df_plot.records_ratio],
            textposition='auto',
            name='Count'
        ),secondary_y=False)

    fig.add_trace(
        go.Scatter(
            x=df_plot[col],
            y=df_plot.fraud_ratio,
            name='Fraud Ratio'
        ),secondary_y=True)

    fig.update_layout(
        title_text="{} Values Distribution and % of Transaction Frauds".format(col_alias[col])
    )

    fig.update_xaxes(title_text=col_alias[col])
    fig.update_yaxes(title_text="# transaction records", secondary_y=False)
    fig.update_yaxes(title_text="% fraud records", secondary_y=True)

    return fig

def plot_time_dist_and_fraud_pct_plotly(df, col):
    df_plot = df.groupby(col).agg(records_num = ('isFraud', 'size'),
                                     records_ratio = ('isFraud',lambda x: np.round(len(x)/len(df)*100, 2)), 
                                     fraud_ratio = ('isFraud', lambda x: np.round(x.mean()*100, 2))).reset_index()
    df_plot = df_plot.sort_values(col)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=df_plot[col],
            y=df_plot.records_num,
            marker={'color': df_plot[col],
                    'colorscale': 'Spectral'},
            text=[str(x)+'%' for x in df_plot.records_ratio],
            textposition='auto',
            name='Count'
        ),secondary_y=False)

    fig.add_trace(
        go.Scatter(
            x=df_plot[col],
            y=df_plot.fraud_ratio,
            name='Fraud Ratio'
        ),secondary_y=True)

    fig.update_layout(
        title_text="{} Values Distribution and % of Transaction Frauds".format(col_alias[col])
    )

    fig.update_xaxes(title_text=col_alias[col])
    fig.update_yaxes(title_text="# transaction records", secondary_y=False)
    fig.update_yaxes(title_text="% fraud records", secondary_y=True)

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)