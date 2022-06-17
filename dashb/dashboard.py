#importing libraries 


import pandas as pd
import plotly.express as px
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input,Output
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import dash_bootstrap_components as dbc


# initializing the dash app and using external CYBORG theme provided by bootstrap 
app = dash.Dash(external_stylesheets=[dbc.themes.CYBORG])


# reading the csv file and storing the data in the pandas dataframe
df = pd.read_csv('data_preprocessed.csv')
df.drop(['Unnamed: 0'],axis=1,inplace=True)
df.drop(['Unnamed: 0.1'],axis=1,inplace=True)


###################################################################################################
# -------------------------------------figure functions--------------------------------------------
###################################################################################################


def top_10_clubs():
    '''
    Function used to return a bar chart representing the top 10 clubs 
    with the highest player overall average along with the player 
    potential average that can be achieved in the future
    '''
    over = df.groupby('Club').mean()['Overall'].to_numpy()
    potential = df.groupby('Club').mean()['Potential'].to_list()
    clubs = df.groupby('Club').mean().index
    # creating data with (club name,club overall , club potential)
    clubdata = list(zip(clubs,over,potential))
    df_clubs = pd.DataFrame(clubdata,columns=['Club','Avg Overall','Avg Potential'])
    df_clubs = df_clubs.sort_values(by='Avg Overall',ascending=False)
    # creating bar chart with both club potential and overall
    fig = go.Figure(data=[
        go.Bar(name='Overall', x=df_clubs['Club'].head(10), y=df_clubs['Avg Overall'].head(10)),
        go.Bar(name='Potential', x=df_clubs['Club'].head(10), y=df_clubs['Avg Potential'].head(10))
    ])
    fig.update_layout(template='plotly_dark')
    
    return fig


def preferred_foot():
    '''
    Function used to return a pie chart representing the preferred foot of the 
    footballers in the dataset
    '''
    # counting players with preferred foot 
    mean_wage = list(df.groupby('Preferred Foot').mean()['Wage'])[::-1]
    index = list(df['Preferred Foot'].value_counts().index)
    count = list(df['Preferred Foot'].value_counts())
    df1 = pd.DataFrame({'foot':index,'count':count,'mean wage':mean_wage})
    fig=px.pie(df1,names = 'foot' , values = 'count',template='plotly_dark',hover_data=['mean wage'])
    
    return fig




def pos_attributes():
    '''
    Function used to sunburst chart representing the most important attributes 
    a player requires in a particular position
    '''
    # index is a list of indices of the column we want for out sunburst graph
    index = [4]
    index.extend(list(range(21,55)))        
    positions = ['GoalKeeper','Defender','Midfielder','Attacker']
    feature=[]
    for pos in positions:
        # selecting the top 5 attributes for each position
        feature.extend(list(df[df['General Position']==pos].iloc[:,index].corr()['Overall'].sort_values(ascending=False).index)[2:7])
    # converting it to a format which can be passed to the sunburst function
    data = {
        'Attribute':feature,
        'Parent':['GoalKeeper']*5+['Defender']*5+['Midfielder']*5+['Attacker']*5,
    }
    # creating the sunburst graph
    fig = px.sunburst(data,path=['Parent','Attribute'],template='plotly_dark')
    return fig



def top_20_countries():
    '''
    Function used to return a geographic scatter plot representing the countries
    with most players in FIFA with their count
    '''

    # creating the dataset with countries and the number of players in the that country
    countries = list(df['Nationality'].value_counts().index)
    num_of_players = list(df['Nationality'].value_counts())
    df_countries = pd.DataFrame({'Country':countries,'num of players':num_of_players})
    df_countries = df_countries.sort_values(by='num of players',ascending=False)
    fig = px.scatter_geo(df_countries.head(20), locations="Country",size='num of players',color='Country',
                     projection="natural earth",locationmode='country names',template='plotly_dark')  
    return fig



def wage_interval():
    '''
    Function used to return a pie chart representing the wage distribution of the 
    footballers in the dataset 
    '''
    interval = ['0k to 5k','5k to 10k','10k to 100k','100k+']
    # counting players who lie in their respective wage interval
    wage = []
    wage.append(len(df[(df['Wage']>=0)&(df['Wage']<5)]))
    wage.append(len(df[(df['Wage']>=5)&(df['Wage']<10)]))
    wage.append(len(df[(df['Wage']>=10)&(df['Wage']<100)]))
    wage.append(len(df[(df['Wage']>=100)]))
    fig = px.pie(df,names=interval,values=wage,template='plotly_dark')
    return fig

def value_interval():
    '''
    Function used to return a pie chart representing the value distribution of the 
    footballers in the dataset 
    '''
    interval=['0k to 500k','500k to 1000k','1000k to 2000k','2000k to 10000k','10000k+']
    # counting players who lie in their respective value interval
    value = []
    value.append(len(df[(df['Value']>=0)&(df['Value']<500)]))
    value.append(len(df[(df['Value']>=500)&(df['Value']<1000)]))
    value.append(len(df[(df['Value']>=1000)&(df['Value']<2000)]))
    value.append(len(df[(df['Value']>=2000)&(df['Value']<10000)]))
    value.append(len(df[(df['Value']>=10000)]))
    fig = px.pie(df,names=interval,values=value,template='plotly_dark')
    return fig

# using a callback to get option parameter from the dropdown and send the histogram to the figure attribute of the graph
@app.callback(
    Output(component_id='histogram',component_property='figure'),
    [Input(component_id='slct_hist',component_property='value')]
)
def update_hist(option):
    '''
    Function used to return a histogram of the value passed in the option parameter
    which is to be selected from the dropdown.
    The dropdown options are->Player Overall
                            ->Player Potential
                            ->Player Age
                            ->Player Position
    '''
    
    fig = px.histogram(df,x=option,color=option,template='plotly_dark')
    return fig

# using callback to get the option parameter from the tabs option provided and send the table to the figure attribute of the graph
@app.callback(
    Output(component_id='player_feature',component_property='figure'),
    [Input(component_id='slct_player_feature',component_property='value')]
)
def update_feature_graph(option):
    '''
    Function used the return a Table depending on the value of the option parameter passed
    which is to be selected from the tabs that have been provided.
    The tab options are -> Best Players
                        -> Best Goalkeeper
                        -> Best Defender
                        -> Best Attacker
                        -> Best Midfielder
                        -> Value for Money (players which are available for less price that they should be)
                        -> Hidden Gems (players that might become the Top players in the future)
    '''

    # if condition used to select which data should be provided to the table


    if(option=='All'):
        fig = go.Figure(data=[go.Table(header={'values':['Player Name','Club','Player Position','Overall'],'line_color':'darkgray','fill_color':'black',
                'font':dict(color='white', size=12)},
                                cells={'values':[df['Name'].head(10),df['Club'].head(10),df['Position'].head(10),df['Overall'].head(10)],
                                'line_color':'darkgray','fill_color':'darkslategray','font':dict(color='white', size=12)})])
        
    elif(option=='Defender'):
        df1 = df[df['General Position']=='Defender']
        fig = go.Figure(data=[go.Table(header={'values':['Player Name','Club','Player Position','Overall'],'line_color':'darkgray','fill_color':'black',
                            'font':dict(color='white', size=12)},
                                cells={'values':[df1['Name'].head(10),df1['Club'].head(10),df1['Position'].head(10),df1['Overall'].head(10)],
                                'line_color':'darkgray','fill_color':'darkslategray','font':dict(color='white', size=12)})])
    
    elif(option=='GoalKeeper'):
        df1 = df[df['General Position']=='GoalKeeper']
        fig = go.Figure(data=[go.Table(header={'values':['Player Name','Club','Player Position','Overall'],'line_color':'darkgray','fill_color':'black',
                        'font':dict(color='white', size=12)},
                                cells={'values':[df1['Name'].head(10),df1['Club'].head(10),df1['Position'].head(10),df1['Overall'].head(10)],
                                'line_color':'darkgray','fill_color':'darkslategray','font':dict(color='white', size=12)})])
    
    
    elif(option=='Midfielder'):
        df1 = df[df['General Position']=='Midfielder']
        fig = go.Figure(data=[go.Table(header={'values':['Player Name','Club','Player Position','Overall'],'line_color':'darkgray','fill_color':'black',
                                'font':dict(color='white', size=12)},
                                cells={'values':[df1['Name'].head(10),df1['Club'].head(10),df1['Position'].head(10),df1['Overall'].head(10)],
                                'line_color':'darkgray','fill_color':'darkslategray','font':dict(color='white', size=12)})])
    

    elif(option=='Attacker'):
        df1 = df[df['General Position']=='Attacker']
        fig = go.Figure(data=[go.Table(header={'values':['Player Name','Club','Player Position','Overall'],'line_color':'darkgray','fill_color':'black',
                            'font':dict(color='white', size=12)},
                                cells={'values':[df1['Name'].head(10),df1['Club'].head(10),df1['Position'].head(10),df1['Overall'].head(10)],
                                'line_color':'darkgray','fill_color':'darkslategray','font':dict(color='white', size=12)})])
    
    elif(option=='Val'):
        # calculating the best players with lower price than expected 
        df1 = df[(df['Value']!=0) & (df['General Position']!='GoalKeeper')].loc[:,['Overall','Value','Name','Club','Position']].copy()
        df1['lvalue'] = np.log(df1['Value'])
        sc = StandardScaler()
        df1['norm_over'] = sc.fit_transform(df1.loc[:,['Overall']])
        df1['lvalue'] = sc.fit_transform(df1.loc[:,['lvalue']])
        df1['lvalue'] = df1.lvalue - df1.lvalue.min() + 1
        df1.norm_over = df1.norm_over - df1.norm_over.min() + 1
        df1.norm_over = df1.norm_over - df1.norm_over.min() + 1
        df1['val for money'] = df1['norm_over']/df1['lvalue']
        df1 = df1[df1['Overall']>75].sort_values(by='val for money',ascending = False)
        fig = go.Figure(data=[go.Table(header={'values':['Player Name','Club','Player Position','Overall'],'line_color':'darkgray','fill_color':'black',
                                'font':dict(color='white', size=12)},
                                cells={'values':[df1['Name'].head(10),df1['Club'].head(10),df1['Position'].head(10),df1['Overall'].head(10)],
                                'line_color':'darkgray','fill_color':'darkslategray','font':dict(color='white', size=12)})])
    
    elif option == 'hid':
        # finding the players with most  difference between their potential and overall
        df['Potential_Diff'] = df['Potential'] - df['Overall']
        df1 = df[df['Potential']>85]
        df1 = df1.sort_values(by='Potential_Diff',ascending=False)
        fig = go.Figure(data=[go.Table(header={'values':['Player Name','Club','Player Position','Overall','Potential'],'line_color':'darkgray','fill_color':'black',
                                'font':dict(color='white', size=12)},
                                cells={'values':[df1['Name'].head(10),df1['Club'].head(10),df1['Position'].head(10),df1['Overall'].head(10),df1['Potential'].head(10)],
                                'line_color':'darkgray','fill_color':'darkslategray','font':dict(color='white', size=12)})])
    
    
    fig.update_layout(template='plotly_dark')
    
    
    return fig

# ---------------------------------figure functions end------------------------------
# -------------------------------------app layout-------------------------------------


# css classes for content and cards

CONTENT_STYLE = {
    'margin-left': '5%',
    'margin-right': '5%',
    'padding': '20px 10p'
}


CARD_TEXT_STYLE = {
    'textAlign': 'center',
    # 'color': '#0074D9'
    'color': '#FFFFFF'
}

# first row of the content containing cards with information about the data
content_first_row = dbc.Row([
    dbc.Col(
        dbc.Card(
            [

                dbc.CardBody(
                    [
                        html.H4(id='card_title_1', children=['Number of PLayers'], className='card-title',
                                style=CARD_TEXT_STYLE),
                        html.P(id='card_text_1', children=[len(df)], style=CARD_TEXT_STYLE),
                    ]
                )
            ]
        ),
        md=3
    ),
    dbc.Col(
        dbc.Card(
            [

                dbc.CardBody(
                    [
                        html.H4('Number of Countries', className='card-title', style=CARD_TEXT_STYLE),
                        html.P(df['Nationality'].nunique(), style=CARD_TEXT_STYLE),
                    ]
                ),
            ]

        ),
        md=3
    ),
    dbc.Col(
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H4('Number of Clubs', className='card-title', style=CARD_TEXT_STYLE),
                        html.P(df['Club'].nunique(), style=CARD_TEXT_STYLE),
                    ]
                ),
            ]

        ),
        md=3
    ),
    dbc.Col(
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H4('Number of Indian Players', className='card-title', style=CARD_TEXT_STYLE),
                        html.P(len(df[df['Nationality']=='India']), style=CARD_TEXT_STYLE),
                    ]
                ),
            ]
        ),
        md=3
    )
])


# second row of the content containing the histogram with the dropdown and the wage distribution pie chart 
content_second_row = dbc.Row(
    [
        dbc.Col(
            [dcc.Dropdown(id='slct_hist',
                options=[
                    {"label":'Age','value':'Age'},
                    {"label":'Player Overall','value':'Overall'},
                    {"label":'Player Potential','value':'Potential'},
                    {"label":'Position','value':'Position'},
                    ],

                value = "Overall"
                ),
            html.Br(),
            dcc.Graph(id='histogram',figure={})],
             md=8
        ),
        dbc.Col(
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H4('Wage Distribution (€)', className='card-title', style=CARD_TEXT_STYLE),
                        dcc.Graph(id='graph',figure=wage_interval()),
                    ]
                ),
            ]
        ),
        md=4
    )
    ]
)


# third row of the content containing the geographic scatter plot and the value distribution pie chart
content_third_row = dbc.Row(
    [
        dbc.Col(
            [html.Br(),
                html.H4('Countries with most players',className='card-title', style=CARD_TEXT_STYLE),
                dcc.Graph(id='graph_4',figure=top_20_countries())], md=8,
        ),
        
        dbc.Col(
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H4('Value Distribution (€)', className='card-title', style=CARD_TEXT_STYLE),
                        dcc.Graph(id='g',figure=value_interval()),
                    ]
                ),
            ]
        ),
        md=4),
        
        
    ]
)


# forth row of the content containing the top clubs histogram and the best attributes sunburst chart
content_fourth_row = dbc.Row(
    [
        dbc.Col(
            [html.H4('Top 10 Clubs (According to Overall)', className='card-title', style=CARD_TEXT_STYLE),
            dcc.Graph(id='graph_5',figure=top_10_clubs())], md=8
        ),
        dbc.Col(
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H4('Best Attributes (According to position)', className='card-title', style=CARD_TEXT_STYLE),
                        dcc.Graph(id='pos_attribute',figure=pos_attributes()),
                    ]
                ),
            ]
        ),
        md=4),
    ]
)


# fifth row of the content containing the age vs potential graph and the preferred foot pie chart
content_fifth_row = dbc.Row(
    [
        dbc.Col(
            [html.Br(),
                html.H4('Age vs Potential',className='card-title', style=CARD_TEXT_STYLE),
                dcc.Graph(id='grap',figure=px.scatter(df,x='Age',y='Potential',trendline='ols',template='plotly_dark'))], md=8,
        ),
        
        dbc.Col(
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H4('Preferred Foot and wage comparison', className='card-title', style=CARD_TEXT_STYLE),
                        dcc.Graph(id='preferred_foot',figure=preferred_foot()),
                    ]
                ),
            ]
        ),
        md=4),
        
        
    ]
)

# sixth row of the content containing the table with the data about players in FIFA 
content_sixth_row = dbc.Row(
    [
        dbc.Col(
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H1('TransferMarket Area', className='card-title', style=CARD_TEXT_STYLE),
                        html.Br(),
                        dcc.Tabs(id='slct_player_feature',value='All',children=[
                        dcc.Tab(label='Best Players',value='All'),
                        dcc.Tab(label='Best Goalkeeper',value='GoalKeeper'),
                        dcc.Tab(label='Best Defenders',value='Defender'),
                        dcc.Tab(label='Best Attacker',value='Attacker'),
                        dcc.Tab(label='Best Midfielder',value='Midfielder'),
                        dcc.Tab(label='Value for Money',value='Val'),
                        dcc.Tab(label='Hidden Gem',value='hid')
                    ]),
                    dcc.Graph(id='player_feature',figure={})
                    ]
                ),
            ]
        ),
        md=12)
    ]
)

# this is the point where the content rows and combined 
content = html.Div(
    [
        html.H2('FIFA Dashboard', style=CARD_TEXT_STYLE),
        html.Hr(),
        content_first_row,
        html.Hr(),
        html.Br(),
        content_second_row,
        html.Hr(),
        html.Br(),
        content_third_row,
        html.Hr(),
        html.Br(),
        content_fourth_row,
        html.Hr(),
        html.Br(),
        content_fifth_row,
        html.Hr(),
        html.Br(),
        content_sixth_row
    ],
    style=CONTENT_STYLE
)

# final app layout with the content 
app.layout = html.Div([content])



# ----------------------------------app layout ends-------------------------------------






if __name__ == '__main__':
    app.run_server(debug=False)


# -------------------------------END---------------------------------------