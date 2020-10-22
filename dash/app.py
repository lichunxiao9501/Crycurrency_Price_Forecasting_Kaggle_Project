# Import required libraries
import os
import pickle
import copy
import datetime as dt
import re

import numpy as np
import pandas as pd
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

app = dash.Dash(__name__)
server = app.server

# Multi-dropdown options

# Create controls 


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
                     #showlegend=False
                     ),
                     )
pie_fig.update_traces(marker=dict(line=dict(color='white', width=2)))
pie_fig.update_layout(title='Cryptocurrency market share (%)')        

df_top_10_cryptos = df[df.name.isin(top_10_cryptos)]
ts_fig = px.line(x = df_top_10_cryptos.index, 
              y = df_top_10_cryptos.close, 
              color = df_top_10_cryptos.name,
              color_discrete_map = dict(zip(labels, colors)),
    labels = {'x':'Date','y':'Close Price','color':'Cryptocurrecy'},
    title = 'Daily close price of top 10 cryptocurrencies')
ts_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                  plot_bgcolor='rgba(0,0,0,0)'
                 )

# Load logo
BTC_LOGO = os.path.join('btc-logo.png')

header = html.Div(
                [
                    html.Img(
                        src=app.get_asset_url(BTC_LOGO),
                        style = {'height':'100px', 'width':'100px'},
                        className='two columns',
                    ),
                    html.Div(
                        [
                            html.H2(
                                'Top Cryptocurrecies',
                            ),
                            html.H4(
                                'Price and Volume Overview',
                            )
                        ],
                        id='title',
                        className='eight columns',
                    ),
                    html.A(
                        html.Button(
                            "Learn More",
                            id="learnMore"

                        ),
                        href="https://github.com/lichunxiao9501",
                        className="two columns"
                    )
                ],
                id="header",
                className='row',
            )

info_boxes = html.Div(
    [
        html.Div([
            html.P("Cryptocurrency"),
            html.H6(
                id="crypto_type",
                className="info_text"
                )
                ],
                className="three columns pretty_container"
                ),
        html.Div(
            [
                html.P("Average Volume"),
                html.H6(
                    id="volumeText",
                    className="info_text"
                    )
            ],
            className="three columns pretty_container"
            ),
        html.Div(
            [
                html.P("Lowest Price"),
                html.H6(
                    id="lowText",
                    className="info_text"
                    )
            ],
            className="three columns pretty_container"
            ),
        html.Div(
            [
                html.P("Highest Price"),
                html.H6(
                    id="highText",
                    className="info_text"
                    )
            ],
            className="three columns pretty_container"
            ),
        ],
        id="infoContainer",
        className="row"
    )

# Create app layout
app.layout = html.Div([
    header,
    html.Div([
                html.Div([
                    html.P(
                            'Filter by cruptocurrency type (clicking pie slide):',
                                className="control_label"
                            ),
                            dcc.Graph(id='pie_graph')
                ],
                className="pretty_container four columns"),
                html.Div([
                    info_boxes,
                    html.Div(
                                [
                                    dcc.Graph(id='ts_graph')
                                ],
                                id="countGraphContainer",
                                className="pretty_container"
                            )
                ],
                id="rightCol",
                className="eight columns"
                ),
            ],
            className="row"
        )
    ],
    id="mainContainer",
    style={
        "display": "flex",
        "flex-direction": "column"
    }
)


# Create callbacks
@app.callback([Output('volumeText', 'children'),
                Output('highText', 'children'),
                Output('lowText', 'children')],
             [Input('crypto_type','children'),
             Input('ts_graph','relayoutData')],
             [State('ts_graph','figure')])
def update_text(selected, ts_graph_relayout, ts_graph):
    if (selected == '-') | (selected == 'Others'):
        return '-', '-', '-'

    start = ts_graph['layout']['xaxis']['range'][0]
    end = ts_graph['layout']['xaxis']['range'][1]
    selected_df = df[(df.name == selected)&(df.index>=start)&(df.index<=end)]
    low = '$'+str(selected_df.close.min())
    high = '$'+str(selected_df.close.max())
    avg_vol = f"{int(selected_df.volume.mean()):,d}"
    return avg_vol,high,low

@app.callback([Output('pie_graph','figure'),
                Output('ts_graph', 'figure'),
                Output('crypto_type', 'children')],
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
            return pie_fig, ts_fig, '-'
        if (len(current_selected) == 1) & (current_selected[0] == selection[0]):
            return pie_fig, ts_fig, '-'
        if (len(current_selected) == len(current_colors)) | (current_selected[0] != selection[0]):
            # Update colors
            new_colors = list(trans_colors)
            new_colors[selection[0]] = colors[selection[0]]
            fig["data"][0]["marker"]["colors"] = new_colors
            # new_ts_fig = plot_volume_and_price2(df, labels[selection[0]], colors[selection[0]], rolling_window = 1)
            new_ts_fig = candlestick_chart(df, labels[selection[0]], increasing_color = colors[selection[0]], decreasing_color = '#7F7F7F')
            return fig, new_ts_fig, labels[selection[0]]
    return pie_fig, ts_fig, '-'

# Helper functions
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

def candlestick_chart(df, crypto, increasing_color = '#17BECF', decreasing_color = '#7F7F7F'):
    '''this function draw the candlestick chart with 10-day period moving average and bollinger band'''
    # Filering selected crypto data
    crypto_df = df[df['name']==crypto].copy()
    crypto_df.drop(['symbol','name','ranknow','slug'],axis=1,inplace=True)
   
    # Adding layout
    layout = {
        'xaxis': {
            'rangeselector': {
                'visible': True
            }
        },
        # Adding a volume bar chart for candlesticks is a good practice usually
        'yaxis': {
            'domain': [0, 0.2],
            'showticklabels': False
        },
        'yaxis2': {
            'domain': [0.2, 0.8]
        },
        'legend': {
            'orientation': 'h',
            'y': 0.9,
            'yanchor': 'bottom'
        },
        'margin': {
            't': 40,
            'b': 40,
            'r': 40,
            'l': 40
        },
        'paper_bgcolor':'rgba(0,0,0,0)',
        'plot_bgcolor':'rgba(0,0,0,0)',
    }

    # Adding some range buttons to interact
    rangeselector = {
        'visible': True,
        'x': 0,
        'y': 0.8,
        'buttons': [
            {'count': 1, 'label': 'reset', 'step': 'all'},
            {'count': 1, 'label': '1 y', 'step': 'year', 'stepmode': 'backward'},
            {'count': 6, 'label': '6 mo', 'step': 'month', 'stepmode': 'backward'},
            {'count': 3, 'label': '3 mo', 'step': 'month', 'stepmode': 'backward'},
            {'count': 1, 'label': '1 mo', 'step': 'month', 'stepmode': 'backward'},
        ]
    }

    layout['xaxis'].update(rangeselector=rangeselector)

    # Adding traces
    data = []
    # Defining main chart
    trace0 = go.Candlestick(
        x=crypto_df.index, 
        open=crypto_df['open'], 
        high=crypto_df['high'],
        low=crypto_df['low'], 
        close=crypto_df['close'],
        yaxis='y2', name=crypto,
        increasing=dict(line=dict(color=increasing_color)),
        decreasing=dict(line=dict(color=decreasing_color)),
    )

    data.append(trace0)


    # Adding volume bar chart
    colors = [increasing_color if x else decreasing_color for x in crypto_df['open'] < crypto_df['close']]

    trace1 = go.Bar(
        x=crypto_df.index, 
        y=crypto_df['volume'],
        marker=dict(color=colors),
        yaxis='y', name='Volume'
    )

    data.append(trace1)

    # Adding Moving Average
    moving_average = crypto_df['close'].rolling(10, center=True).mean().dropna()

    trace2 = go.Scatter(
        x=moving_average.index, 
        y=moving_average,
        yaxis='y2', name='Moving Average',
        line=dict(width=1)
    )

    data.append(trace2)

    # Adding boilinger bands
    def bollinger_bands(price, window_size=10, num_of_std=1.5):
        rolling_mean = price.rolling(10, center=True).mean().dropna()
        rolling_std = price.rolling(10, center=True).std().dropna()
        upper_band = rolling_mean + (rolling_std * 1.5)
        lower_band = rolling_mean - (rolling_std * 1.5)
        return upper_band, lower_band

    bb_upper, bb_lower = bollinger_bands(crypto_df['close'])

    trace3 = go.Scatter(
        x=bb_upper.index, 
        y=bb_upper,
        yaxis='y2', 
        line=dict(width=1),
        marker=dict(color='#ccc'), 
        hoverinfo='none',
        name='Bollinger Bands',
        legendgroup='Bollinger Bands'
    )
    data.append(trace3)

    trace4 = go.Scatter(
        x=bb_lower.index, 
        y=bb_lower,
        yaxis='y2', 
        line=dict(width=1),
        marker=dict(color='#ccc'), 
        hoverinfo='none',
        name='Bollinger Bands', showlegend=False,
        legendgroup='Bollinger Bands'
    )
    data.append(trace4)

    fig = go.Figure(data=data, layout=layout)
    return fig

# Main
if __name__ == '__main__':
    app.server.run(debug=True, threaded=True)
