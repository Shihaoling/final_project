import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

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

plt.style.use('seaborn')

# read the file and do the cleanning in advanced file(it only lacks little data)
regular = pd.read_csv('2021-2022 NBA Player Stats - Regular.csv',delimiter=";", encoding="latin-1", index_col=0)
advanced = pd.read_excel('2021-2022advanced.xlsx',index_col=0)
advanced['TS%'].fillna(advanced['TS%'].mean(),inplace=True)
advanced['3PAr'].fillna(advanced['3PAr'].mean(),inplace=True)
advanced['FTr'].fillna(advanced['FTr'].mean(),inplace=True)
advanced['TOV%'].fillna(advanced['TOV%'].mean(),inplace=True)
play_by_play = pd.read_excel("2021-2022play-by-play.xlsx",index_col=0,header=1)

# build the app
selected_season = st.sidebar.radio(label='Select the season', options=pd.Series(["2021-2022", '2020-2021', '2019-2020']), index=0)

#delete the TOT and use the multiselect
unique_team = regular['Tm'].unique()
unique_team = np.delete(unique_team,np.where(unique_team == 'TOT'))
selected_team = st.sidebar.multiselect('Select the team', unique_team, unique_team)

searched_name = st.text_input(label='Search for player:',value='')

# Judge wether the name is in the dataset
if searched_name != '' and not regular[regular['Player'].isin([searched_name])].empty:
    regular = regular[regular.Player == searched_name]
elif searched_name != '' and regular[regular['Player'].isin([searched_name])].empty:
    possible_find = findsimilar(regular,searched_name)
    st.subheader(f'We cannot find the player you provide. Do you find {possible_find}?')
    regular = regular[regular.Player == possible_find]
    searched_name = possible_find

# filter according to the team.
regular = regular[regular['Tm'].isin(selected_team)]
st.write(regular)
# Draw the plots
if searched_name =='':
    for team in selected_team:
        st.write(f'Main indicators of {team} in Season 2021-2022')
        fig, ax = plt.subplots()
        pd.Series(list(regular[regular['Tm'] == team]['PTS']),index=list(regular[regular['Tm'] == team]['Player'])).sort_values(ascending=False).plot.bar(ax=ax)
        st.pyplot(fig)