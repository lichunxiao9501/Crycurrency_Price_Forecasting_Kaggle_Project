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
import re
import os
import pickle

from textwrap import dedent as d
import json

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# the style arguments for the sidebar.
styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

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


# Load data
df = pd.read_csv('../data/crypto-markets.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace = True)
current_market = df[(df.index.year==2018)&(df.index.month==11)].groupby(['symbol','name','ranknow']).agg(market = ('market', 'mean')).reset_index()
current_market.sort_values(['market','ranknow'], ascending=False, inplace=True)
current_market['total_market_perc'] = np.round(current_market.market.cumsum() / current_market.market.sum() * 100, 2)
top_10_cryptos = current_market.name.head(10).values.tolist()

btc = pd.read_csv('../output/btc.csv')
btc['date'] = pd.to_datetime(btc['date'])
btc.set_index('date', inplace=True)
n = len(btc)
date_dict = dict(zip(range(n), btc.index))

# resmapling to monthly frequency
btc_month = btc.resample('M').mean()
btc_month['log_close'] = np.log(btc_month.close)

# resmapling to weekly frequency
btc_week = btc.resample('W-MON').mean()
btc_week['log_close'] = np.log(btc_week.close)

# create pie plot for top 10 cryptos
current_market_top_10 = current_market.copy()
current_market_top_10.loc[~current_market_top_10.name.isin(top_10_cryptos), 'name'] = 'Others'
current_market_top_10 = current_market_top_10.groupby('name')['market'].sum().reset_index()
values = current_market_top_10.market
labels = current_market_top_10.name
colors = px.colors.qualitative.Set3[:len(current_market_top_10)]
trans_colors = ['rgba('+re.split("[\(\)]",c)[1]+',0.2)' for c in colors]

pie_fig = go.Figure()
pie_fig.add_trace(go.Pie(
                     values = values,
                     labels = labels,
                     marker_colors=colors,
                     hovertext = np.round(current_market_top_10.market / 1000000000, 2),
                     hovertemplate = "Crypto:%{label} <br>Market: $%{hovertext}B",
                     showlegend=False),
                     )
pie_fig.update_traces(marker=dict(line=dict(color='white', width=2)))
pie_fig.update_layout(title='Percentage of cryptocurrency market')        

df_top_10_cryptos = df[df.name.isin(top_10_cryptos)]
ts_fig = px.line(x = df_top_10_cryptos.index, 
              y = df_top_10_cryptos.close, 
              color = df_top_10_cryptos.name,
              color_discrete_map = dict(zip(top_10_cryptos, colors)),
    labels = {'x':'Date','y':'Close Price','color':'Cryptocurrecy'},
    title = 'Daily close price of top 10 cryptocurrencies')
ts_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                  plot_bgcolor='rgba(0,0,0,0)'
                 )


# Load logo
BTC_LOGO = os.path.join('btc-logo.png')


controls = dbc.FormGroup(
    [
        html.P('Cryptocurrency Type', style={
            'textAlign': 'center'
        }),
        
        dcc.Dropdown(
                        id='cryptos',
                        options=[{'label': x, 'value':x} for x in df.name.unique()],
                        value=top_10_cryptos, # default value
                        multi=True
                    ),
        html.Br(),

        html.P('Date Range:', style={
            'textAlign': 'center'
        }),
        dcc.RangeSlider(
                            id='date_slider',
                            min=0,
                            max=n-1,
                            value=[0, n-1],
                            # marks=btc.date,
                        ),
        html.Br(),              
        html.P('Forecast Start Date:', style={
                    'textAlign': 'center'
                }),
        dcc.Dropdown(
                            id='test_date',
                            options=[{'label': x.strftime("%b, %Y"), 'value':x} for x in btc_month.index],
                            value=pd.to_datetime('2018-1-31') # default value
                        ),

        html.Br(),

        # dbc.Button(
        #     id='submit_button',
        #     n_clicks=0,
        #     children='Submit',
        #     color='primary',
        #     block=True
        # ),
    ]
)

# sidebar
sidebar = html.Div(
    [
        html.H2('Crypto Filters', style=TEXT_STYLE),
        html.Hr(),
        controls
    ],
    style=SIDEBAR_STYLE,
)

content_zero_row = dbc.Row([
            dbc.Col(
                dcc.Graph(id='cryptos_graph'), md=12
                )
        ])

content_first_row = dbc.Row([
            dbc.Col(
                dcc.Graph(id='pie_graph'), md=4
                ),
            dbc.Col(
                dcc.Graph(id='ts_graph'), md=8
                )
        ])

content_second_row = dbc.Row([
            dbc.Col(
                dcc.Graph(id='arima_graph'), md=12
                ),
            # dbc.Col(
            #     dcc.Graph(id='ts_graph'), md=6
            #     )
        ])

content_third_row = html.Div(className='row', children=[
        html.Div([
            dcc.Markdown(d("""
                **Hover Data**

                Mouse over values in the graph.
            """)),
            html.Pre(id='hover-data', style=styles['pre'])
        ], className='six columns'),

        html.Div([
            dcc.Markdown(d("""
                **Click Data**

                Click on points in the graph.
            """)),
            html.Pre(id='click-data', style=styles['pre']),
        ], className='six columns')
 ])

content = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(html.Img(src=app.get_asset_url(BTC_LOGO), style = {'height':'100px'}), md=2),
                dbc.Col(html.H2('Bitcoin Daily Price Dashboard', style=TEXT_STYLE), md=8),
                dbc.Col(md=2),
            ],
            align="center",
            no_gutters=True,
            ),
        html.Hr(),
        content_zero_row,
        content_first_row,
        content_second_row,
        html.Hr(),
        # dbc.Row([
        #     html.H4('Fraud Ratio by Time')
        # ],
        # style = {'margin-top':'20px', 'margin-bottom':'50px'}),
        content_third_row,
        # content_fourth_row
    ],
    style=CONTENT_STYLE
)

app.layout = html.Div([sidebar, content])

# Create callbacks

# @app.callback(Output('ts_graph', 'figure'),
#               [Input('date_slider', 'value')])
# def make_ts_figure(date_slider):
#     # colors = []
#     # for i in range(1960, 2018):
#     #     if i >= int(year_slider[0]) and i < int(year_slider[1]):
#     #         colors.append('rgb(123, 199, 255)')
#     #     else:
#     #         colors.append('rgba(123, 199, 255, 0.2)')
#     start = date_dict[date_slider[0]]
#     end = date_dict[date_slider[1]]
#     return plot_volume_and_price(btc, start, end, rolling_window = 1)

@app.callback([Output('pie_graph','figure'),
                Output('ts_graph', 'figure')],
             [Input('pie_graph','clickData')],
             [State('pie_graph','figure')])
def update_color(clickData, fig):
    selection = None
    # Update selection based on which event triggered the update.
    trigger = dash.callback_context.triggered[0]["prop_id"]
    if trigger == 'pie_graph.clickData':
        current_colors = list(fig["data"][0]["marker"]["colors"])
        current_selected = [i for i, color in enumerate(current_colors) if color.rsplit('(')[0]=='rgb']
        selection = [point["pointNumber"] for point in clickData["points"]]
        if len(current_selected) == 0:
            return pie_fig, ts_fig
        if (len(current_selected) == 1) & (current_selected[0] == selection[0]):
            return pie_fig, ts_fig
        if (len(current_selected) == len(current_colors)) | (current_selected[0] != selection[0]):
            # Update colors
            new_colors = list(trans_colors)
            new_colors[selection[0]] = colors[selection[0]]
            fig["data"][0]["marker"]["colors"] = new_colors
            new_ts_fig = plot_volume_and_price2(df, labels[selection[0]], colors[selection[0]], rolling_window = 1)
            return fig, new_ts_fig
    return pie_fig, ts_fig


@app.callback(
    Output('hover-data', 'children'),
    [Input('pie_graph', 'hoverData')])
def display_hover_data(hoverData):
    return json.dumps(hoverData, indent=2)


@app.callback(
    Output('click-data', 'children'),
    [Input('pie_graph', 'clickData')])
def display_click_data(clickData):
    return json.dumps(clickData, indent=2)

@app.callback(Output('arima_graph', 'figure'),
              [Input('test_date', 'value')])
def make_time_series_figure(test_date):
    fig = arima_after_a_month_forecast(test_date, btc_month)
    return fig

@app.callback(Output('cryptos_graph', 'figure'),
              [Input('cryptos', 'value'),
              Input('date_slider', 'value')]
              )
def cryptos_figure(cryptos, date_slider):
    start = date_dict[date_slider[0]]
    end = date_dict[date_slider[1]]
    fig = plot_cryptos(cryptos, start, end)
    return fig

# Helper functions
def plot_volume_and_price(df,  start = None, end = None, rolling_window = 1):
    '''this function plot volume and high and low price within a given time frame'''
    # rolling average
    df = df[['open','high','low','close','volume','market']].rolling(window=rolling_window).mean().dropna()

    date = selected_df.index.to_series()
    high = selected_df.high
    low = selected_df.low
    close = selected_df.close
    vol = selected_df.volume
    
    # plot
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=pd.concat([date,date[::-1]]),
        y=pd.concat([high,low[::-1]]),
        fill='toself',
        fillcolor='rgba(231,107,243,0.2)',
        line_color='rgba(255,255,255,0)',
        showlegend=False,
        name='Price',
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=date,
        y=close,
        line_color='rgb(231,107,243)',
        showlegend=False,
        name='Price',
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=date,
        y=vol,
        fill='tozeroy',
        fillcolor='rgba(189,189,189,0.5)',
        line_color='rgba(189,189,189,0)',
        showlegend=False,
    ), secondary_y=True)


    fig.update_layout(title='Volume and High and Low Price of Bitcoin',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      yaxis2_title='Volume',
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)'
                    )
    
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1M",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6M",
                         step="month",
                         stepmode="backward"),
                    dict(count=1,
                         label="YTD",
                         step="year",
                         stepmode="todate"),
                    dict(count=1,
                         label="1Y",
                         step="year",
                         stepmode="backward"),
                    dict(label= 'All',
                         step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )

    return fig

def plot_volume_and_price2(df, crypto, color, rolling_window = 1):
    '''this function plot volume and high and low price within a given time frame'''
    # extract selected crypto
    crypto_df = df[df['name']==crypto].copy()

    # drop irrelavant columns
    crypto_df.drop(['symbol','name','ranknow','slug'],axis=1,inplace=True)

    # rolling average
    crypto_df = crypto_df[['open','high','low','close','volume','market']].rolling(window=rolling_window).mean().dropna()

    # prepare data
    date = crypto_df.index.to_series()
    high = crypto_df.high
    low = crypto_df.low
    close = crypto_df.close
    vol = crypto_df.volume
    
    # plot
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=pd.concat([date,date[::-1]]),
        y=pd.concat([high,low[::-1]]),
        fill='toself',
        fillcolor='rgba('+re.split("[\(\)]",color)[1]+',0.2)',
        line_color='rgba(255,255,255,0)',
        showlegend=False,
        name='Price',
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=date,
        y=close,
        line_color=color,
        showlegend=False,
        name='Price',
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=date,
        y=vol,
        fill='tozeroy',
        fillcolor='rgba(189,189,189,0.5)',
        line_color='rgba(189,189,189,0)',
        showlegend=False,
    ), secondary_y=True)


    fig.update_layout(title='Daily high, low, close prices and volume of {}'.format(crypto),
                      xaxis_title='Date',
                      yaxis_title='Price',
                      yaxis2_title='Volume',
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)'
                    )
    
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1M",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6M",
                         step="month",
                         stepmode="backward"),
                    dict(count=1,
                         label="YTD",
                         step="year",
                         stepmode="todate"),
                    dict(count=1,
                         label="1Y",
                         step="year",
                         stepmode="backward"),
                    dict(label= 'All',
                         step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )

    return fig

def arima_after_a_month_forecast(test_start_date = '2018-1-31', btc_month = btc_month):
    '''this function fits arima models for the walk forward forecast of the close price and plots the results'''
    X = np.log(btc_month.close)
    test_dates = X.index[X.index>=test_start_date]

    pred_mean = []
    pred_lower = []
    pred_upper = []
    for t in test_dates:
        # past data end date
        t0 = X.index[X.index.get_loc(t)-1]

        # fit arima model
        model = SARIMAX(X[:t0], order=(1, 1, 0)).fit()
        btc_month_dynamic = model.get_prediction(start=t, end=t, dynamic=True, full_results=True)
        pred_mean.append(np.exp(btc_month_dynamic.predicted_mean.values[0]))

        # take 95% confidence interval 
        pred_dynamic_ci = btc_month_dynamic.conf_int(alpha=0.05)
        pred_lower.append(np.exp(pred_dynamic_ci['lower close'].values[0]))
        pred_upper.append(np.exp(pred_dynamic_ci['upper close'].values[0]))

    # create prediciton df
    pred_df = pd.DataFrame({'date':test_dates, 'forecast':pred_mean, 'lower':pred_lower, 'upper':pred_upper})
    pred_df['actual'] = btc_month.close[pred_df['date'].values].values

    # plot forecast and 95% CI
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=btc_month.index, y=btc_month.close, name='Actual'))
    fig.add_trace(go.Scatter(x=pred_df.date, y=pred_df.lower, showlegend=False, marker_color='lightgrey'))
    fig.add_trace(go.Scatter(x=pred_df.date, y=pred_df.upper, name = '95% CI', marker_color='lightgrey', fill='tonexty'))
    fig.add_trace(go.Scatter(x=pred_df.date, y=pred_df.forecast, name='ARIMA Forecast', marker_color='red'))
    fig.update_layout(title='Bitcoin monthly close price forecast',
                          xaxis_title='Date',
                          yaxis_title='Close price (USD)',
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)'
                        )
    
    return fig

def plot_cryptos(selcted_cryptos, start, end):
    if selcted_cryptos==top_10_cryptos:
        fig_title = 'Daily close price of top 10 cryptocurrencies'
    else:
        fig_title = 'Daily close price of selected cryptocurrencies'
    df_selcted_cryptos = df[df.name.isin(selcted_cryptos)&(df.index>=start)&(df.index<=end)]
    fig = px.line(x = df_selcted_cryptos.index, y = df_selcted_cryptos.close, color = df_selcted_cryptos.name,
        labels = {'x':'Date','y':'Close Price','color':'Cryptocurrecy'},
        title = fig_title)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)'
                      )
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)