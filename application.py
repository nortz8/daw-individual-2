import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import pandas as pd
from sklearn import model_selection
import pickle
from sklearn.ensemble import GradientBoostingRegressor
import plotly.graph_objects as go

app = dash.Dash(__name__)
app.scripts.config.serve_locally = True

# Load Model
filename = './data/mrt_model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))

dfx = pd.read_csv('./data/Testing_Data.csv')
X2 = dfx.drop('entry', axis=1)
y2 = dfx['entry']

# Predict The Number of Passengers on the Test Data
y_predict = loaded_model.predict(X2)
Final_Data = X2
Final_Data['Actual'] = y2
Final_Data['Predicted'] = y_predict

# Change NaN and Negative Value Predictions to Zero
Final_Data.loc[(Final_Data['Predicted'] < 0) | (Final_Data['Predicted'].isnull()), 'Predicted'] = 0

# Save as csv for backup
Final_Data.to_csv('./data/Final_Data.csv',index=False)

# Create dataframes saved as csv for all stations for plotting
dfs = Final_Data
for n in range(1,14):
    df1 = dfs[dfs['station']==n]
    df1.to_csv('./data/Station_'+str(n)+'.csv')


app.layout = html.Div([
    html.Center(html.H1(children='MRT Sakay: Predicting Hourly Number of Passengers in an MRT-3 Station',style={'color': 'black'})),
    html.Div(html.Center(html.H2(children='Machine Learning Project',style={'color':'black'}))),
    html.Div(html.Center(html.H3(children='Term 2 - Learning Team 7',style={'color':'black'}))),
    html.Div(html.Center(html.H4(children='MSDS 2020',style={'color':'black'}))),

    html.Div(html.H3(children='Abstract',style={'color': 'black'})),

    html.Div(html.H4(children='Predicting the number of passengers in the crowded MRT-3 is an important first step in planning future improvements and in managing foot traffic in the train stations. In this study, we used Logistic Regression (L1 and L2), Random Forest Regression, and Gradient Boosted Regression to predict the hourly passenger count in an MRT train station. Historical passenger data for 2016 and 2017 was collected from the Department of Transportation (DoTr) official website and weather data is from wundergound.com. The Gradient Boosted Model with learning rate of 0.2 and max depth of 20 produced the best result with an average r-squared accuracy of 0.9436 using 10-fold cross validation.',style={'color': 'black'})),

    html.Div(html.P([html.Br(style={'color': 'black'})])),
    html.Div(html.H3(children='Prediction',style={'color': 'black'})),
    html.Div(html.H4(children='Below is a one-day hourly passenger count prediction from 4AM to 11PM.',style={'color':'black'})),
    html.Div(html.H4(children='Choose Train Station Below:',style={'color': 'black'})),


    dcc.Dropdown(
        id='station',
        options=[
            {'label': 'North Avenue', 'value': './data/Station_1.csv'},
            {'label': 'Quezon Avenue', 'value': './data/Station_2.csv'},
            {'label': 'GMA Kamuning', 'value': './data/Station_3.csv'},
            {'label': 'Araneta Center Cubao', 'value': './data/Station_4.csv'},
            {'label': 'Santolan Annapolis', 'value': './data/Station_5.csv'},
            {'label': 'Ortigas', 'value': './data/Station_6.csv'},
            {'label': 'Shaw Boulevard', 'value': './data/Station_7.csv'},
            {'label': 'Boni', 'value': './data/Station_8.csv'},
            {'label': 'Guadalupe', 'value': './data/Station_9.csv'},
            {'label': 'Buendia', 'value': './data/Station_10.csv'},
            {'label': 'Ayala', 'value': './data/Station_11.csv'},
            {'label': 'Magallanes', 'value': './data/Station_12.csv'},
            {'label': 'Taft Avenue', 'value': './data/Station_13.csv'}
        ],
        value='./data/Station_4.csv',
        clearable=False
        ),
        
    html.Div(html.P([html.Br(style={'color': 'black'})])),
    dcc.Graph(id='Plot'),
    html.Div(html.P([html.Br(style={'color': 'black'})])),

    html.Div(html.H3(children='Recommendations',style={'color': 'black'})),
    html.Div(html.H4(children='The study focused on models available using scikit-learn. As an extension of this study, deep learning models could be explored to find out if these can give more accurate hourly passenger count predicitons. Passenger historical data available from the Department of Transportation is also limited to the number tap-ins. It is suggested to coordinate with DoTr or other agencies if they could include an estimate on the number of passengers outside the station and in the lines before the tap-in machines to improve the usability of this model.',style={'color':'black'})),

    ],
      style={
        'borderBottom': 'black',
        'backgroundColor': 'black',
        'padding': '30px',
        'background-image': 'url(https://docs.google.com/uc?export=view&id=1oXK2tXrB1zj0wvwMcfiRq_0SdTHoPyOf)'}
    ) 

@app.callback(
    dash.dependencies.Output('Plot', 'figure'),
    [dash.dependencies.Input('station','value')]
    )
def plotting(station):
    dfzz = pd.read_csv(station)
    x = dfzz['time']
    y= dfzz['Actual']
    yhat= dfzz['Predicted']
    x_rev = x[::-1]

    # Line - Actual Values
    y1 = y
    y1_upper = y1*1.001
    y1_lower = y1*0.999
    y1_lower = y1_lower[::-1]

    # Line - Predicted Value
    y2 = yhat
    y2_upper = y2*1.01
    y2_lower = y2*0.99
    y2_lower = y2_lower[::-1]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x.tolist()+x_rev.tolist(),
        y=y1_upper.tolist()+y1_lower.tolist(),
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line_color='rgba(255,255,255,0)',
        showlegend=False,
        name='Actual',
    ))

    fig.add_trace(go.Scatter(
        x=x.tolist()+x_rev.tolist(),
        y=y2_upper.tolist()+y2_lower.tolist(),
        fill='toself',
        fillcolor='rgba(231,107,243,0.2)',
        line_color='rgba(255,255,255,0)',
        showlegend=False,
        name='Actual',
    ))

    fig.add_trace(go.Scatter(
        x=x, y=y1,
        line_color='rgb(0,100,80)',
        name='Actual',
    ))

    fig.add_trace(go.Scatter(
        x=x, y=y2,
        line_color='rgb(231,107,243)',
        name='Predicted',
    ))

    fig.update_traces(mode='lines')
    return fig

application = app.server
if __name__ == '__main__':
    app.run_server(port=8082)
