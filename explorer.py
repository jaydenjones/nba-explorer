import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests
import sys
import ssl 
import certifi 
from urllib.request import urlopen 

st.title('NBA Player Stats Explorer')

st.markdown("""
This app performs simple webscraping of NBA player stats data!
* **Python libraries:** base64, pandas, streamlit
* **Data source:** [Basketball-reference.com](https://www.basketball-reference.com/).
""")

st.sidebar.header('User Input Features')
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1950,2023))))

# Web scraping of NBA player stats
@st.cache_data
def load_data(year):
    url = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_per_game.html"
    html = urlopen(url, context=ssl.create_default_context(cafile=certifi.where())) # just so I didn't get SSL cert errors; better than just ignoring it I guess
    html_list = pd.read_html(html)
    html_list = html_list[0] 
    df = pd.DataFrame(html_list)
    raw = df.drop(df[df.Age == 'Age'].index) # Deletes repeating headers in content
    raw = raw.fillna(0)
    playerstats = raw.drop(['Rk'], axis=1)
    return playerstats
playerstats = load_data(selected_year)


# team - selection
sorted_unique_team = sorted(playerstats['Tm'].unique()) # FIXME
selected_team = st.sidebar.multiselect('Team', sorted_unique_team, sorted_unique_team) # third arg is default value

# position selection
unique_pos = ['C', 'PF', 'SF', 'SG', 'PG']
selected_pos = st.sidebar.multiselect('Position', unique_pos, unique_pos) # third arg is default value

# filtering data
df_selected_team = playerstats[(playerstats['Tm'].isin(selected_team)) & (playerstats['Pos'].isin(selected_pos))]

st.header('Display Player Stats of Selected Team(s)')
st.write(f'Data Dimension:{str(df_selected_team.shape[0])} rows and {str(df_selected_team.shape[1])} columns' )
st.dataframe(df_selected_team)

# download csv
def file_download(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode() # strings <--> bytes
    href = f'<a href="data:file/csv;base64,{b64}" download="player_stats.csv">Download CSV File</a>'
    return href

st.markdown(file_download(df_selected_team), unsafe_allow_html=True)

# Heatmap
if st.button('Intercorrelation Heatmap'):
    st.header('Intercorrelation Heatmap')
    df_selected_team.to_csv('output.csv', index=False)
    df = pd.read_csv('output.csv') # annoying but works

    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style('white'):
        fig, ax = plt.subplots(figsize=(7,5))
        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
        st.pyplot(fig)

