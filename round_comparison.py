import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import requests
import json
import utils
from datetime import datetime, timezone

st.set_page_config(
    page_title="Round Comparison",
    page_icon="📊",
    layout="wide",

)

st.title('⚖️ Gitcoin Grants Round Comparison')

st.title("Comparing Crowdfunding Dollars ($) By Day of Round")
df = utils.get_beta_votes_data()
df_alpha = pd.read_csv('alpha_by_day.csv')
df_alpha.drop(['date'], axis=1, inplace=True) 
# strip $ and ,
df_alpha['amountUSD'] = df_alpha['amountUSD'].str.replace('$', '')
df_alpha['amountUSD'] = df_alpha['amountUSD'].str.replace(',', '')
df_alpha['amountUSD'] = df_alpha['amountUSD'].astype(float)


if "data_loaded" in st.session_state and st.session_state.data_loaded:
    dfv = st.session_state.dfv
    dfp = st.session_state.dfp
    round_data = st.session_state.round_data
else:
    data_load_state = st.text('Loading data...')
    dfv, dfp, round_data = utils.load_round_data()
    data_load_state.text("")


dfv = dfv.drop_duplicates()
# group by date of timestamp and round_name, get sum of amountUSD and count of rows
dfv_grouped = dfv.groupby([pd.Grouper(key='timestamp', freq='D'), 'round_name']).agg({'amountUSD': 'sum', 'voter': 'count'}).reset_index()
# rename voter to votes 
dfv_grouped.rename(columns={'voter': 'votes'}, inplace=True)
dfv_grouped['day_number'] = dfv_grouped.groupby('round_name').cumcount() + 1
# drop timestamp 
dfv_grouped.drop(['timestamp'], axis=1, inplace=True)
dfv_grouped['program'] = "GG18"


# concat dfv_grouped and df
df_all = pd.concat([dfv_grouped, df])
# order values by program, round_name, day_number
df_all.sort_values(by=['program', 'day_number','round_name' ], inplace=True)


# Replace comma in the amount_usd column and convert it to a float
df_all['amountUSD'] = df_all['amountUSD'].astype(float)

# Compute cumulative sum for each program
df_all['cumulative_amount'] = df_all.groupby('program')['amountUSD'].cumsum()
df_grouped = df_all.groupby([ 'program', 'day_number'])['cumulative_amount'].max().reset_index()
df_alpha['cumulative_amount'] = df_alpha.groupby('program')['amountUSD'].cumsum()
df_grouped = pd.concat([df_grouped, df_alpha[['program', 'day_number', 'cumulative_amount']]])


col1, col2 = st.columns(2)
# Plot using Plotly
fig = px.line(df_grouped, x='day_number', y='cumulative_amount', color='program', title='Cumulative sum of amountUSD by day and program', labels={'day_number': 'Day', 'cumulative_amount': 'Cumulative Amount (in dollars)'}, color_discrete_map={'GG18': 'red', 'Beta': 'blue', 'Alpha': 'brown'})
col1.plotly_chart(fig, use_container_width=True)

# Pivot table to get day as index, program as columns and cumulative_amount as values
pivot_df = df_grouped.pivot(index='day_number', columns='program', values='cumulative_amount')

# Compute % difference for each day between GG18 and other rounds based on the max value
pivot_df['percentage_difference_beta'] = ((pivot_df['GG18'] - pivot_df['Beta']) / pivot_df['Beta']) * 100
pivot_df['percentage_difference_alpha'] = ((pivot_df['GG18'] - pivot_df['Alpha']) / pivot_df['Alpha']) * 100


# Plot the % differences
fig = px.bar(pivot_df.reset_index(), x='day_number', y=[ 'percentage_difference_alpha', 'percentage_difference_beta'], 
             title='% Difference between Prior Rounds and GG18 based on Cumulative Amount Raised by Day',
             labels={'value': 'Percentage Difference', 'variable': 'Program'})
col2.plotly_chart(fig, use_container_width=True)

# Compute the latest valid percentage difference for Beta
idx_beta = -1
pct_diff_beta = pivot_df['percentage_difference_beta'].iloc[idx_beta]
while np.isnan(pct_diff_beta):
    idx_beta -= 1
    pct_diff_beta = pivot_df['percentage_difference_beta'].iloc[idx_beta]

# Compute the latest valid percentage difference for Alpha
idx_alpha = -1
pct_diff_alpha = pivot_df['percentage_difference_alpha'].iloc[idx_alpha]
while np.isnan(pct_diff_alpha):
    idx_alpha -= 1
    pct_diff_alpha = pivot_df['percentage_difference_alpha'].iloc[idx_alpha]

# Round the percentage differences to 2 decimal places
pct_diff_beta = round(pct_diff_beta, 2)
pct_diff_alpha = round(pct_diff_alpha, 2)

# Display messages for both programs
if pct_diff_beta < 0: 
    st.subheader(f"GG18 crowdfunding (in dollars) is currently {pct_diff_beta}% lower than the same time of Beta")
else:
    st.subheader(f"GG18 crowdfunding (in dollars) is currently {pct_diff_beta}% higher than the same time of Beta")

if pct_diff_alpha < 0: 
    st.subheader(f"GG18 crowdfunding (in dollars) is currently {pct_diff_alpha}% lower than the same time of Alpha")
else:
    st.subheader(f"GG18 crowdfunding (in dollars) is currently {pct_diff_alpha}% higher than the same time of Alpha")

# Formatting for displaying pivot_df
pivot_df['percentage_difference_beta'] = pivot_df['percentage_difference_beta'].map('{:,.2f}%'.format)
pivot_df['percentage_difference_alpha'] = pivot_df['percentage_difference_alpha'].map('{:,.2f}%'.format)
st.write(pivot_df)

st.title("Comparing Crowdfunding Contributions (#) By Day of Round")
# Compute cumulative sum for votes for each program
df_all['cumulative_votes'] = df_all.groupby('program')['votes'].cumsum()
df_votes_grouped = df_all.groupby(['program', 'day_number'])['cumulative_votes'].max().reset_index()
df_alpha['cumulative_votes'] = df_alpha.groupby('program')['votes'].cumsum()
df_votes_grouped = pd.concat([df_votes_grouped, df_alpha[['program', 'day_number', 'cumulative_votes']]])


# Pivot table to get day as index, program as columns and cumulative_votes as values
pivot_df_votes = df_votes_grouped.pivot(index='day_number', columns='program', values='cumulative_votes')

# Compute % difference for each day between GG18 and other rounds based on the max value
pivot_df_votes['percentage_difference_votes_beta'] = ((pivot_df_votes['GG18'] - pivot_df_votes['Beta']) / pivot_df_votes['Beta']) * 100
pivot_df_votes['percentage_difference_votes_alpha'] = ((pivot_df_votes['GG18'] - pivot_df_votes['Alpha']) / pivot_df_votes['Alpha']) * 100

col1, col2 = st.columns(2)
# Plot using Plotly
fig = px.line(df_votes_grouped, x='day_number', y='cumulative_votes', color='program', title='Cumulative Number of Contributions by day and program', labels={'day_number': 'Day', 'cumulative_amount': 'Cumulative Number of Donations'}, color_discrete_map={'GG18': 'red', 'Beta': 'blue', 'Alpha': 'brown'})
col1.plotly_chart(fig, use_container_width=True)

# Plot the % differences for votes
fig_votes = px.bar(pivot_df_votes.reset_index(), x='day_number', y=['percentage_difference_votes_alpha', 'percentage_difference_votes_beta'], 
                   title='% Difference between Prior Rounds and GG18 based on Cumulative Votes by Day',
                   labels={'value': 'Percentage Difference', 'variable': 'Program'})
col2.plotly_chart(fig_votes, use_container_width=True)

# Compute the latest valid percentage difference for votes for Beta
idx_votes_beta = -1
pct_diff_votes_beta = pivot_df_votes['percentage_difference_votes_beta'].iloc[idx_votes_beta]
while np.isnan(pct_diff_votes_beta):
    idx_votes_beta -= 1
    pct_diff_votes_beta = pivot_df_votes['percentage_difference_votes_beta'].iloc[idx_votes_beta]

# Compute the latest valid percentage difference for votes for Alpha
idx_votes_alpha = -1
pct_diff_votes_alpha = pivot_df_votes['percentage_difference_votes_alpha'].iloc[idx_votes_alpha]
while np.isnan(pct_diff_votes_alpha):
    idx_votes_alpha -= 1
    pct_diff_votes_alpha = pivot_df_votes['percentage_difference_votes_alpha'].iloc[idx_votes_alpha]

# Round the percentage differences for votes to 2 decimal places
pct_diff_votes_beta = round(pct_diff_votes_beta, 2)
pct_diff_votes_alpha = round(pct_diff_votes_alpha, 2)

# Display messages for both programs based on votes
if pct_diff_votes_beta < 0: 
    st.subheader(f"GG18 crowdfunding votes are currently {pct_diff_votes_beta}% lower than the same time of Beta")
else:
    st.subheader(f"GG18 crowdfunding votes are currently {pct_diff_votes_beta}% higher than the same time of Beta")

if pct_diff_votes_alpha < 0: 
    st.subheader(f"GG18 crowdfunding votes are currently {pct_diff_votes_alpha}% lower than the same time of Alpha")
else:
    st.subheader(f"GG18 crowdfunding votes are currently {pct_diff_votes_alpha}% higher than the same time of Alpha")

# Formatting for displaying pivot_df_votes
pivot_df_votes['percentage_difference_votes_beta'] = pivot_df_votes['percentage_difference_votes_beta'].map('{:,.2f}%'.format)
pivot_df_votes['percentage_difference_votes_alpha'] = pivot_df_votes['percentage_difference_votes_alpha'].map('{:,.2f}%'.format)
st.write(pivot_df_votes)

max_alpha_votes = pivot_df_votes['Alpha'].max()
max_beta_votes = pivot_df_votes['Beta'].max()
max_gg18_votes = pivot_df_votes['GG18'].max()

max_alpha_amount = pivot_df['Alpha'].max()
max_beta_amount = pivot_df['Beta'].max()
max_gg18_amount = pivot_df['GG18'].max()

col1, col2, col3 = st.columns(3)
col1.subheader("Alpha")
col1.metric("Votes", f"{max_alpha_votes:,}" )
col1.metric("Amount", f"${max_alpha_amount:,.2f} ")
col1.metric("Amount per Vote", f"${max_alpha_amount/max_alpha_votes:,.2f} ")
col2.subheader("Beta")
col2.metric("Votes", f"{max_beta_votes:,}" )
col2.metric("Amount", f"${max_beta_amount:,.2f} ")
col2.metric("Amount per Vote", f"${max_beta_amount/max_beta_votes:,.2f} ")
col3.subheader("GG18")
col3.metric("Votes", f"{max_gg18_votes:,}" )
col3.metric("Amount", f"${max_gg18_amount:,.2f} ")
col3.metric("Amount per Vote", f"${max_gg18_amount/max_gg18_votes:,.2f} ")

# Set the target time: August 29th, 2023 at 12 PM UTC
target_time = datetime(2023, 8, 29, 12, 0, tzinfo=timezone.utc)
time_left = utils.get_time_left(target_time)
col3.subheader("")
col3.subheader("⏰ Time Left:")
col3.subheader((time_left))
