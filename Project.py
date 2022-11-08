import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import plotly.figure_factory as ff
import seaborn as sns
import scipy
import base64


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

def findsimilar(data,searched_name):
    letter_vector = [0]*26
    for letter in searched_name.lower():
        if ord(letter)<=96 or ord(letter)>= 123:
            continue
        position = ord(letter) - 96
        if letter_vector[position-1] == 0:
            letter_vector[position-1] = 1
    total_letter_vector = []
    for i in data['Player']:
        vector = [0]*26
        i = i.lower()
        for j in i:
            if ord(j) <= 96 or ord(j) >= 123:
                continue
            position = ord(j) - 96
            if vector[position-1] == 0:
                vector[position-1] = 1
        total_letter_vector.append(vector)
    min = 100
    index = 0
    position = 0
    for i in total_letter_vector:
        distance = np.linalg.norm(np.array(i) - np.array(letter_vector))
        if distance < min:
            min = distance
            position = index
        index += 1
    return data['Player'].iloc[position]


add_bg_from_local('background.png')
plt.style.use('seaborn')

st.spinner(text="We will show detail you want soon......")

# do the description work
st.header('Welcome to NBA Data World')
st.markdown('In our websites, you can freely browse data for players in the league. You can also query data using different conditions.  Affiliated with the data you want, we also provide **insightgul and professional** analysis.')

# read the file and do the cleanning in advanced file(it only lacks little data)
basic = pd.read_excel('2021-2022basic.xlsx', index_col=0)
basic_mirror = basic
advanced = pd.read_excel('2021-2022advanced.xlsx',index_col=0)
play_by_play = pd.read_excel("2021-2022play-by-play.xlsx",index_col=0,header=1)
total = pd.read_excel("2021-2022total.xlsx",index_col=0)

# Deal with the data
data1 = total[['Player','Pos','Age','Tm','G','MP','FG%','3P','3PA','3P%','2P','2PA','FT','FTA','PTS']]

# Use the tab function to make the app more orderly.
tab_basic, tab_3PTS, tab_3PTA, tab_free_throw = st.tabs(["Basic Data", "Total scores and three point scores","three point percentage and attempts","Free throw Analysis"])

with tab_basic:

    # build the app
    selected_season = st.sidebar.radio(label='Select the season', options=pd.Series(["2021-2022"]), index=0)

    #delete the TOT and use the multiselect
    unique_team = basic['Tm'].unique()
    unique_team = np.delete(unique_team,np.where(unique_team == 'TOT'))
    selected_team = st.sidebar.multiselect('Select the team', unique_team, unique_team)

    searched_name = st.text_input(label='Search for player:',value='')

    #get the data we need from the whole dataset first, Then we will use metric function to show the detail information about individual
    # Judge wether the name is in the dataset and deal with it
    st.subheader('Player info')
    if searched_name != '' and not basic[basic['Player'].isin([searched_name])].empty:
        individual_3PT = basic[basic.Player == searched_name]['3P%'].mean()
        delt_individual_3PT = individual_3PT - basic['3P%'].mean()
        individual_2PT = basic[basic.Player == searched_name]['2P%'].mean()
        delt_individual_2PT = individual_2PT - basic['2P%'].mean()
        individual_PTS = basic[basic.Player == searched_name]['PTS'].mean()
        delt_individual_PTS = individual_PTS - basic['PTS'].mean()
        basic = basic[basic.Player == searched_name]
        index_3 = int(np.where(basic_mirror.sort_values(by='3P%')['Player'] == searched_name)[0][0]) + 1
        percent_3 = index_3 / len(basic_mirror['Player'])
        index_2 = int(np.where(basic_mirror.sort_values(by='2P%')['Player'] == searched_name)[0][0]) + 1
        percent_2 = index_2 / len(basic_mirror['Player'])
        index_T = int(np.where(basic_mirror.sort_values(by='PTS')['Player'] == searched_name)[0][0]) + 1
        percent_T = index_T / len(basic_mirror['Player'])
        col1, col2, col3 = st.columns(3)
        col1.metric(f'Individual 3P% ({percent_3:.2%})', f'{individual_3PT:.2%}',f'{delt_individual_3PT:.2%}')
        col2.metric(f'Individual 2P% ({percent_2:.2%})', f'{individual_2PT:.2%}',f'{delt_individual_2PT:.2%}')
        col3.metric(f'Individual PTS ({percent_T:.2%})', f'{individual_PTS:.2f}'+' PTS',f'{delt_individual_PTS:.2f} PTS')

    elif searched_name != '' and basic[basic['Player'].isin([searched_name])].empty:
        searched_name = findsimilar(basic,searched_name)
        individual_3PT = basic[basic.Player == searched_name]['3P%'].mean()
        delt_individual_3PT = individual_3PT - basic['3P%'].mean()
        individual_2PT = basic[basic.Player == searched_name]['2P%'].mean()
        delt_individual_2PT = individual_2PT - basic['2P%'].mean()
        individual_PTS = basic[basic.Player == searched_name]['PTS'].mean()
        delt_individual_PTS = individual_PTS - basic['PTS'].mean()
        st.subheader(f'We cannot find the player you provide. Do you find {searched_name}?')
        basic = basic[basic.Player == searched_name]
        index_3 = int(np.where(basic_mirror.sort_values(by='3P%')['Player'] == searched_name)[0][0]) + 1
        percent_3 = index_3 / len(basic_mirror['Player'])
        index_2 = int(np.where(basic_mirror.sort_values(by='2P%')['Player'] == searched_name)[0][0]) + 1
        percent_2 = index_2 / len(basic_mirror['Player'])
        index_T = int(np.where(basic_mirror.sort_values(by='PTS')['Player'] == searched_name)[0][0]) + 1
        percent_T = index_T / len(basic_mirror['Player'])
        col1, col2, col3 = st.columns(3)
        col1.metric(f'Individual 3P% ({percent_3:.2%})', f'{individual_3PT:.2%}',f'{delt_individual_3PT:.2%}')
        col2.metric(f'Individual 2P% ({percent_2:.2%})', f'{individual_2PT:.2%}',f'{delt_individual_2PT:.2%}')
        col3.metric(f'Individual PTS ({percent_T:.2%})', f'{individual_PTS:.2f}'+' PTS',f'{delt_individual_PTS:.2f} PTS')

    # filter according to the team. and do the explanation work
    basic = basic[basic['Tm'].isin(selected_team)]
    st.write(basic)
    expander = st.expander("See explanation")
    expander.write(f"""
    The percentage in the parenthesis means the percentage player exceeds in total NBA players
    \nRk -- Rank
    \nPos -- Position
    \nAge -- Player's age on February 1 of the season
    \nTm -- Team
    \nG -- Games
    \nGS -- Games Started
    \nMP -- Minutes Played Per Game
    \nFG -- Field Goals Per Game
    \nFGA -- Field Goal Attempts Per Game
    \nFG% -- Field Goal Percentage
    \n3P -- 3-Point Field Goals Per Game
    \n3PA -- 3-Point Field Goal Attempts Per Game
    \n3P% -- 3-Point Field Goal Percentage
    \n2P -- 2-Point Field Goals Per Game
    \n2PA -- 2-Point Field Goal Attempts Per Game
    \n2P% -- 2-Point Field Goal Percentage
    \neFG% -- Effective Field Goal Percentage
    \nThis statistic adjusts for the fact that a 3-point field goal is worth one more point than a 2-point field goal.
    \nFT -- Free Throws Per Game
    \nFTA -- Free Throw Attempts Per Game
    \nFT% -- Free Throw Percentage
    \nORB -- Offensive Rebounds Per Game
    \nDRB -- Defensive Rebounds Per Game
    \nTRB -- Total Rebounds Per Game
    \nAST -- Assists Per Game
    \nSTL -- Steals Per Game
    \nBLK -- Blocks Per Game
    \nTOV -- Turnovers Per Game
    \nPF -- Personal Fouls Per Game
    \nPTS -- Points Per Game
    """)

    # Draw the plots
    progress = 0
    st.subheader('Teams info')
    basic = basic_mirror
 
    if searched_name == '':
        if st.button('Show me selected teams info (It may takes long time)'):
            for team in selected_team:
                st.write(f'Main indicators of {team} in Season 2021-2022')
                team_3PT = basic[basic.Tm == team]['3P%'].mean()
                delt_team_3PT = team_3PT - basic['3P%'].mean()
                team_2PT = basic[basic.Tm == team]['2P%'].mean()
                delt_team_2PT = team_2PT - basic['2P%'].mean()
                team_PTS = basic[basic.Tm == team]['PTS'].mean()
                delt_team_PTS = team_PTS - basic['PTS'].mean()
                col1, col2, col3 = st.columns(3)
                col1.metric('Team 3P%', f'{team_3PT:.2%}',f'{delt_team_3PT:.2%}')
                col2.metric('Team 2P%', f'{team_2PT:.2%}',f'{delt_team_2PT:.2%}')
                col3.metric('Team PTS', f'{team_PTS:.2f}'+' PTS',f'{delt_team_PTS:.2f} PTS')

                # PTS plot part
                col_plot, col_data = st.columns([2,1])
                with col_plot:
                    fig, ax = plt.subplots(figsize=(10,6))
                    # img = plt.imread(team + '.png')
                    # ax.imshow(img,extent=[5,21,5,20])
                    pd.Series(list(basic[basic['Tm'] == team]['PTS']),index=list(basic[basic['Tm'] == team]['Player'])).sort_values(ascending=False).plot.bar(ax=ax)
                    ax.set_title(f'PTS of each player in {team}')
                    st.pyplot(fig)
                    
                with col_data:
                    st.write(basic[basic['Tm'] == team][['Player','PTS']].sort_values('PTS',ascending=False).reset_index(drop=True))

                # TRB plot part
                col_plot, col_data = st.columns([2,1])
                with col_plot:
                    fig, ax = plt.subplots(figsize=(10,6))
                    pd.Series(list(basic[basic['Tm'] == team]['TRB']),index=list(basic[basic['Tm'] == team]['Player'])).sort_values(ascending=False).plot.bar(ax=ax)
                    ax.set_title(f'TRB of each player in {team}')
                    st.pyplot(fig)
                with col_data:
                    st.write(basic[basic['Tm'] == team][['Player','TRB']].sort_values('TRB',ascending=False).reset_index(drop=True))

                # AST plot part
                col_plot, col_data = st.columns([2,1])
                with col_plot:
                    fig, ax = plt.subplots(figsize=(10,6))
                    pd.Series(list(basic[basic['Tm'] == team]['AST']),index=list(basic[basic['Tm'] == team]['Player'])).sort_values(ascending=False).plot.bar(ax=ax)
                    ax.set_title(f'AST of each player in {team}')
                    st.pyplot(fig)
                    expander = st.expander(f"Players wordcloud for {team}")
                    expander.image('image/'+team+'.png')
                with col_data:
                    st.write(basic[basic['Tm'] == team][['Player','AST']].sort_values('AST',ascending=False).reset_index(drop=True))

                # 
                # player = '!'.join(regular[regular['Tm'] == team]['Player'].tolist())
                # wordcloud = WordCloud(mask = np.array(Image.open('basketball.jpg')),background_color="white",max_words=100,colormap='Blues',stopwords='!',width=100/120,height=100/120).generate(player)
                # fig, ax = plt.subplots(figsize=(10,10))
                # ax = wordcloud
                # fig.dpi = 100
                # plt.imshow(wordcloud, interpolation="bessel")
                # plt.axis("off")
                # plt.margins(x=0, y=0)
                # st.pyplot(fig)

                # distribution and expander
                PTS_distribution_team = basic[basic['Tm'] == team]['PTS']
                PTS_distribution_total = basic['PTS']
                hist_data = [PTS_distribution_team, PTS_distribution_total]
                group_labels = [f'{team}','Average']
                fig = ff.create_distplot(
                hist_data, group_labels)
                st.plotly_chart(fig)
                expander = st.expander("See explanation")
                expander.write(f"""
                The chart above shows the distribution of PTS in team {team} and the distribution of that on average.(The average PTS each team got)
                """)

    else:
        team_info = basic[basic.Tm == basic[basic['Player'] == searched_name]['Tm'].tolist()[0]]
        team = basic[basic['Player'] == searched_name]['Tm'].tolist()[0]
        st.write(f'Main indicators of {team} in Season 2021-2022')
        team_3PT = basic[basic.Tm == team]['3P%'].mean()
        delt_team_3PT = team_3PT - basic['3P%'].mean()
        team_2PT = basic[basic.Tm == team]['2P%'].mean()
        delt_team_2PT = team_2PT - basic['2P%'].mean()
        team_PTS = basic[basic.Tm == team]['PTS'].mean()
        delt_team_PTS = team_PTS - basic['PTS'].mean()
        col1, col2, col3 = st.columns(3)
        col1.metric('Team 3P%', f'{team_3PT:.2%}',f'{delt_team_3PT:.2%}')
        col2.metric('Team 2P%', f'{team_2PT:.2%}',f'{delt_team_2PT:.2%}')
        col3.metric('Team PTS', f'{team_PTS:.2f}'+' PTS',f'{delt_team_PTS:.2f} PTS')
       # PTS plot part
        col_plot, col_data = st.columns([2,1])
        with col_plot:
            fig, ax = plt.subplots(figsize=(10,6))
            # img = plt.imread(team + '.png')
            # ax.imshow(img,extent=[5,21,5,20])
            pd.Series(list(basic[basic['Tm'] == team]['PTS']),index=list(basic[basic['Tm'] == team]['Player'])).sort_values(ascending=False).plot.bar(ax=ax)
            ax.set_title(f'PTS of each player in {team}')
            st.pyplot(fig)
        with col_data:
            st.write(basic[basic['Tm'] == team][['Player','PTS']].sort_values('PTS',ascending=False).reset_index(drop=True))

        # TRB plot part
        col_plot, col_data = st.columns([2,1])
        with col_plot:
            fig, ax = plt.subplots(figsize=(10,6))
            pd.Series(list(basic[basic['Tm'] == team]['TRB']),index=list(basic[basic['Tm'] == team]['Player'])).sort_values(ascending=False).plot.bar(ax=ax)
            ax.set_title(f'TRB of each player in {team}')
            st.pyplot(fig)
        with col_data:
            st.write(basic[basic['Tm'] == team][['Player','TRB']].sort_values('TRB',ascending=False).reset_index(drop=True))

        # AST plot part
        col_plot, col_data = st.columns([2,1])
        with col_plot:
            fig, ax = plt.subplots(figsize=(10,6))
            pd.Series(list(basic[basic['Tm'] == team]['AST']),index=list(basic[basic['Tm'] == team]['Player'])).sort_values(ascending=False).plot.bar(ax=ax)
            ax.set_title(f'AST of each player in {team}')
            st.pyplot(fig)
            expander = st.expander(f"Players wordcloud for {team}")
            expander.image('image/'+team+'.png')
        with col_data:
            st.write(basic[basic['Tm'] == team][['Player','AST']].sort_values('AST',ascending=False).reset_index(drop=True))

        # distribution and the expander
        PTS_distribution_team = basic[basic['Tm'] == team]['PTS'].reset_index(drop=True)
        PTS_distribution_total = basic['PTS']
        hist_data = [PTS_distribution_team, PTS_distribution_total]
        group_labels = [f'{team}','Average']
        fig = ff.create_distplot(
        hist_data, group_labels)
        st.plotly_chart(fig)
        expander = st.expander("See explanation")
        expander.write(f"""
        The chart above shows the distribution of PTS in team {team} and the distribution of that on average.(The average PTS each team got)
        """)

# Analysis for 3PT part1
with tab_3PTS:
    data1 = data1[data1['Tm'].isin(selected_team)]
    data1 = data1[data1['PTS']>200]
    data1['PTA'] = data1['PTS']/data1['G']
    #PTA的含义是points average：场均得分
    data1['MPA'] = data1['MP']/data1['G']  #MPA场均出场时间
    data1['%_ofPTS_by3P'] = 3*(data1['3P']/data1['PTS'])  #三分球得分占比
    st.subheader('Total points and 3 field goal points')
    PTS_constraints = st.slider('drag the sidebar to assign the lowest total scores',200,2200,200)
    three_P_constraints = st.slider('drag the side bar to assign the lowest number of three points hitted',0,300,0)
    fig, ax = plt.subplots()
    data1 = data1.reset_index(drop = True)
    total_mirror = data1[(data1['PTS'] > PTS_constraints) & (data1['3P'] > three_P_constraints)]
    sns.scatterplot(x = 'PTS', y = 3*data1['3P'], data = total_mirror, hue = 'Pos')
    plt.axvline(x = PTS_constraints ,ls = 'dashed',color = [0,0,0.75])
    plt.axhline(y = 3 * three_P_constraints ,ls = 'dotted',color = [0,0.5,0.5])
    plt.xlim(0,2200)
    plt.ylim(0,900)
    plt.legend(loc = 2, bbox_to_anchor = (1,1))
    st.pyplot(fig)
    st.write(total_mirror[['Player','Tm','PTS','3P','MPA','PTA']])
    expander = st.expander("See explanation")
    expander.write(f"""
    Tm：Team 
    \nPTS：Total points scored in regular season 2021-2022
    \n3P: total three point ball hitted
    \nMPA: average minutes played
    \nPTA: average points
    """) 

    st.subheader('Proportion of 3P Points in Total Points')
    percent_3PT = st.slider('Drag the sidebar to select the least proportion of points by three point ball',0.0,1.0,0.0)
    filter = data1[data1['%_ofPTS_by3P'] > percent_3PT]
    fig, ax = plt.subplots()
    sns.scatterplot(x = 'PTS', y = 3*filter['3P'], data = filter, hue = 'Pos')
    st.pyplot(fig)
    st.write(filter[['Player','Tm','PTS','%_ofPTS_by3P','MPA','PTA','3P','3P%']])
    expander = st.expander("See explanation")
    expander.write(f"""
    Tm：Team 
    \nPTS：Total points scored in regular season 2021-2022
    \n3P: total three point ball hitted
    \nMPA: average minutes played
    \nPTA: average points
    \n3P%: three point field goal percentage
    \n%_ofPTS_by3P: percentage of total points scored by three point ball
    """) 

# Analysis for 3PT part2
with tab_3PTA:
    st.subheader('3-point Field Goal Attempts and 3-point Points Proportion of Total')
    data1 = total[['Player','Pos','Age','Tm','G','MP','FG%','3P','3PA','3P%','2P','2PA','FT','FTA','PTS']]
    data1 = data1[data1['Tm'].isin(selected_team)]
    three_attempts = st.slider('Drag the sidebar to select the least number of three points attempts in regular season 2021-2022',0,800,0)
    three_3P_propotion = st.slider('Drag the sidebar to select the lowest three points field goal percentage in regular season 2021-2022 ',0.0,1.0,0.0)
    data1 = data1.reset_index(drop = True)
    total_mirror = data1[(data1['3PA'] > three_attempts) & (data1['3P%'] > three_3P_propotion)]
    fig, ax = plt.subplots()
    sns.scatterplot(x = '3PA', y = total_mirror['3P'], data = total_mirror, hue = 'Pos')
    plt.legend(loc = 2, bbox_to_anchor = (1,1))
    st.pyplot(fig)
    st.write(total_mirror.drop(['2P','2PA','FT','FTA'], axis = 1))
    expander = st.expander("See explanation")
    expander.write(f"""
    Tm：Team 
    \nPos -- Position
    \nAge -- Player's age on February 1 of the season
    \nTm -- Team
    \nG – Games played in total
    \nMP -- Minutes Played in total
    \nFG% -- Field Goal Percentage
    \n3P -- 3-Point Field Goals Per Game
    \n3PA -- 3-Point Field Goal Attempts Per Game
    \n3P% -- 3-Point Field Goal Percentage
    \nPTS -- Points Per Game
    """) 

# Analysis for free throw
with tab_free_throw:
    basic = basic[basic['Tm'].isin(selected_team)]
    data1 = data1[data1['Tm'].isin(selected_team)]
    st.subheader('FT and FTA with mean data of different position')
    basic.drop(basic[basic['Tm']=='TOT'].index,inplace=True)
    fig, ax = plt.subplots()
    sns.scatterplot(x = 'FTA', y = 'FT', data = basic, hue = 'Pos')
    plt.axvline(x = basic[basic['Pos']=='C']['FTA'].mean(),ls = '--',label = "C'sFTA average",color = [0,0,0.9])
    plt.axvline(x = basic[basic['Pos']=='PG']['FTA'].mean(),ls = '--',label = "PG'sFTA average",color = [0.3,0.3,0.3])
    plt.axvline(x = basic[basic['Pos']=='SG']['FTA'].mean(),ls = '--',label = "SG'sFTA average",color = 'y')
    plt.axvline(x = basic[basic['Pos']=='PF']['FTA'].mean(),ls = '--',label = "PF'sFTA average",color = [0,0.6,0.3])
    plt.axvline(x = basic[basic['Pos']=='SF']['FTA'].mean(),ls = '--',label = "SF'sFTA average",color = 'r')

    plt.axhline(y = basic[basic['Pos']=='C']['FT'].mean(),ls = '--',label = "C'sFT average",color = [0,0,0.9])
    plt.axhline(y = basic[basic['Pos']=='PG']['FT'].mean(),ls = '--',label = "PG'sFT average",color = [0.3,0.3,0.3])
    plt.axhline(y = basic[basic['Pos']=='SG']['FT'].mean(),ls = '--',label = "SG'sFT average",color = [0,0.6,0.6])
    plt.axhline(y = basic[basic['Pos']=='PF']['FT'].mean(),ls = '--',label = "PF'sFT average",color = [0,0.6,0.3])
    plt.axhline(y = basic[basic['Pos']=='SF']['FT'].mean(),ls = '--',label = "SF'sFT average",color = [0,0.9,0])
    plt.legend(loc = 2, bbox_to_anchor = (1,1))
    st.pyplot(fig)
    st.markdown('**Summary**: The vertical lines represent the average Free throw attempts by different positions players, while the horizontal lines represent the average Free throws by different positions players.It reveals that SG(shooting guard) and SF(small forward) are the least two positions in the number of FTA(Free throw average) and FT(Free throw).')

    FTA_number = st.slider('Drag  the sidebar to select the least number of the average free throw attempts in one game', 0, 11,0)
    FT_PTS = st.slider('Drag the sidebar to select then average lowest points got from free throw in one game',0,8,0)
    fig, ax =plt.subplots()
    sns.scatterplot(x = 'FTA', y = 'FT', data = basic[(basic['FTA'] > FTA_number) & (basic['FT'] > FT_PTS)], hue = 'Pos')
    st.pyplot(fig)
    st.write(basic[(basic['FTA'] > FTA_number) & (basic['FT'] > FT_PTS)][['Player','Pos','Tm','FT','FTA','FT%','PTS']])
    expander = st.expander("See explanation")
    expander.write(f"""
    Tm：Team 
    \nPos -- Position
    \nTm -- Team
    \nFT -- Free Throws Per Game
    \nFTA -- Free Throw Attempts Per Game
    \nFT% -- Free Throw Percentage
    \nPTS -- Points Per Game
    """) 

    # Free Throw Percentage
    st.subheader('Free Throw Percentage')
    st.write('We have filter the data for you, data with Games player more than 30 and minutes per play more than 15 mins will be shown.')
    free_throw_percentage = st.slider('Adjust Free Throw Percentage',0.0,1.0,0.0)
    fig, ax = plt.subplots()
    sns.scatterplot(x = 'FTA', y = 'FT', data = basic[(basic['FT%']>free_throw_percentage) & (basic['G']>30) & (basic['MP']>15)], hue = 'Pos')
    plt.legend(loc = 2, bbox_to_anchor = (1,1))
    st.pyplot(fig)
    st.write(basic[(basic['FT%']>free_throw_percentage) & (basic['G']>30) & (basic['MP']>15)][['Player','Pos','Age','Tm','G','MP','FG%', 'eFG%', 'FT','FTA','PTS']])
    expander = st.expander("See explanation")
    expander.write(f"""
    Tm：Team 
    \nPos -- Position
    \nAge -- Player's age on February 1 of the season
    \nTm -- Team
    \nG -- Games
    \nMP -- Minutes Played Per Game
    \nFG% -- Field Goal Percentage
    \neFG% -- Effective Field Goal Percentage
    \nThis statistic adjusts for the fact that a 3-point field goal is worth one more point than a 2-point field goal.
    \nFT -- Free Throws Per Game
    \nFTA -- Free Throw Attempts Per Game
    \nFT% -- Free Throw Percentage
    \nPTS -- Points Per Game
    """) 

    # Percentage of Free Throw
    data1 = total[['Player','Pos','Age','Tm','G','MP','FG%','3P','3PA','3P%','2P','2PA','FT','FTA','PTS']]
    data1 = data1[data1['PTS']>200]
    data1 = data1[data1['Tm'].isin(selected_team)]
    data1['PTA'] = data1['PTS']/data1['G']
    data1['MPA'] = data1['MP']/data1['G']
    data1['FT%'] = data1['FT']/data1['FTA']
    data1['%_ofPTSbyFT'] = data1['FT']/data1['PTS']
    st.subheader('Percentage of Free Throw')
    fig, ax = plt.subplots()
    p1 = sns.scatterplot(x = 'PTS', y = 'FT', data = data1, hue = 'Pos')
    p1.set_title("Free throw scores vs total scores")
    plt.legend(loc = 2, bbox_to_anchor = (1,1))
    st.pyplot(fig)
    st.write('The condition given below is using \'or\'')
    FT_PTS_2 = st.slider('Drag the sidebar to select the least number of total free throws hitted',0,700,0)
    free_throw_percentage_2 = st.slider('Drag the sidebar to select the lowest free throw percentage',0.0,1.0,0.0)
    fig, ax = plt.subplots()
    plt.legend(loc = 2, bbox_to_anchor = (1,1))
    sns.scatterplot(x = 'PTS', y = 'FT', data = data1[(data1['%_ofPTSbyFT'] > free_throw_percentage_2) | (data1['FT'] > FT_PTS_2)], hue = 'Pos')
    st.pyplot(fig)
    st.write(data1[(data1['%_ofPTSbyFT'] > free_throw_percentage_2) | (data1['FT'] > FT_PTS_2)][['Player','Pos','Tm','MPA','FT','FTA','FT%','PTS','PTA','%_ofPTSbyFT']])
    expander = st.expander("See explanation")
    expander.write(f"""
    Tm：Team 
    \nPos -- Position
    \nTm -- Team
    \nMPA: Minutes Played Per Game
    \nFT -- total Free Throws
    \nFTA -- total Free Throw Attempts
    \nFT% -- Free Throw Percentage
    \nPTS -- Total points scored in the season
    \nPTA -- Points Per Game
    \n%_ofPTSbyFT: percentage of points scored by free throw
    """) 
    
    # Total points and percentage of free throw
    st.subheader('Total points and percentage of free throw')
    fig, ax = plt.subplots()
    sns.scatterplot(x = 'PTS', y = '%_ofPTSbyFT', data = data1, hue = 'Pos')
    plt.legend(loc = 2, bbox_to_anchor = (1,1))
    plt.axhline(y = data1[data1['Pos']=='C']['%_ofPTSbyFT'].mean() ,ls = 'dotted',label = 'FT percent for C',color = 'r')
    plt.axhline(y = data1[data1['Pos']=='PG']['%_ofPTSbyFT'].mean() ,ls = 'dashed',label = 'FT percent for PG',color = 'g')
    plt.axhline(y = data1[data1['Pos']=='SG']['%_ofPTSbyFT'].mean() ,ls = 'solid',label = 'FT percent for SG',color = 'y')
    plt.axhline(y = data1[data1['Pos']=='PF']['%_ofPTSbyFT'].mean() ,ls = 'dashdot',label = 'FT percent for PF',color = [0,0,1])
    plt.axhline(y = data1[data1['Pos']=='SF']['%_ofPTSbyFT'].mean() ,ls = ':',label = 'FT percent for SF',color = [0,0.75,0.75])
    plt.legend(loc = 2, bbox_to_anchor = (1,1))
    st.pyplot(fig)
    st.markdown('**Summary**: We can derive the conclusion that Centers rely more on free throws than any other position. This is partly owing to their attacking region being proximate to the basket.')

# end
st.write(' ')
st.write(' ') 
st.write(' ')
st.caption('This is an mini-app designed and made by Haoling Shi and Boyue Yang. student ID: 42054001, 42054055')
st.caption('We use the data from the website [basketball reference](https://www.basketball-reference.com/leagues/NBA_2022_totals.html)')
st.caption('Aurthor\'s Email address: henryhlshi@gmail.com')
st.caption('Copyright © 2022, SWUFE')

