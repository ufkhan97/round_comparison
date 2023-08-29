import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timezone


BASE_URL = "https://indexer-grants-stack.gitcoin.co/data"
time_to_live = 1800  # 30 minutes

# Helper function to load data from URLs
def safe_get(data, *keys):
    """Safely retrieve nested dictionary keys."""
    temp = data
    for key in keys:
        if isinstance(temp, dict) and key in temp:
            temp = temp[key]
        else:
            return None
    return temp

### COME BACK AND FIX THIS
def load_data_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        return response.json()
    except requests.RequestException as e:
        st.warning(f"Failed to fetch data from {url}. Error: {e}")
        return []

@st.cache_resource(ttl=time_to_live)
def load_data(chain_id, round_id, data_type):
    url = f"{BASE_URL}/{chain_id}/rounds/{round_id}/{data_type}.json"
    return load_data_from_url(url)

def transform_projects_data(data):
    projects = []
    for project in data:
        title = safe_get(project, 'metadata', 'application', 'project', 'title')
        grantAddress = safe_get(project, 'metadata', 'application', 'recipient')
        description = safe_get(project, 'metadata', 'application', 'project', 'description')
        
        if title and grantAddress:  # Ensure required fields are available
            project_data = {
                'projectId': project['projectId'],
                'title': title,
                'grantAddress': grantAddress,
                'status': project['status'],
                'amountUSD': project['amountUSD'],
                'votes': project['votes'],
                'uniqueContributors': project['uniqueContributors'],
                'description': description
            }
            projects.append(project_data)
    return projects

@st.cache_resource(ttl=time_to_live)
def load_passport_data():
    url = f"{BASE_URL}/passport_scores.json"
    data = load_data_from_url(url)
    
    passports = []
    for passport in data:
        address = passport.get('address')
        last_score_timestamp = passport.get('last_score_timestamp')
        status = passport.get('status')
        rawScore = safe_get(passport, 'evidence', 'rawScore') or 0

        if address:  # Ensure the required field is available
            passport_data = {
                'address': address,
                'last_score_timestamp': last_score_timestamp,
                'status': status,
                'rawScore': rawScore
            }
            passports.append(passport_data)

    df = pd.DataFrame(passports)
    if not df.empty:
        df['rawScore'] = df['rawScore'].astype(float)
    #df['last_score_timestamp'] = pd.to_datetime(df['last_score_timestamp'])
    return df

def compute_timestamp(row, starting_time, chain_starting_blocks):
    # Get the starting block for the chain_id
    starting_block = chain_starting_blocks[row['chain_id']]
    # Determine the increment based on the chain_id
    increment = 12.0 if row['chain_id'] == 1 else 2
    # Calculate the timestamp based on the blockNumber and starting block
    timestamp = starting_time + pd.to_timedelta((row['blockNumber'] - starting_block) * increment, unit='s')
    # make dt
    return pd.to_datetime(timestamp)

@st.cache_resource(ttl=time_to_live)
def load_round_data(csv_path='gg18_rounds.csv'):
    round_data = pd.read_csv(csv_path)
    #round_data = round_data[round_data['program'] == program]

    dfv_list = []
    dfp_list = []

    for _, row in round_data.iterrows():
        raw_projects_data = load_data(str(row['chain_id']), str(row['round_id']), "applications")
        projects_list = transform_projects_data(raw_projects_data)
        dfp = pd.DataFrame(projects_list)
        dfv = pd.DataFrame(load_data(str(row['chain_id']), str(row['round_id']), "votes"))

        dfp['round_id'] = row['round_id']
        dfp['chain_id'] = row['chain_id']
        dfp['round_name'] = row['round_name']
        dfp['program'] = row['program']

        dfv['round_id'] = row['round_id']
        dfv['chain_id'] = row['chain_id']
        dfv['round_name'] = row['round_name']
        dfv['program'] = row['program']

        dfv_list.append(dfv)
        dfp_list.append(dfp)

    dfv = pd.concat(dfv_list)
    dfp = pd.concat(dfp_list)

    dfp = dfp[dfp['status'] == 'APPROVED']
    

    token_map = {
        "0x0000000000000000000000000000000000000000": "ETH",
        "0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1": "DAI",
    }
    dfv["token_symbol"] = dfv["token"].map(token_map)

    chain_starting_blocks = dfv.groupby('chain_id')['blockNumber'].min().to_dict()
    starting_time = pd.to_datetime(round_data['starting_time'].min())
    dfv['timestamp'] = dfv.apply(compute_timestamp, args=(starting_time, chain_starting_blocks), axis=1)
    dfv['voter'] = dfv['voter'].str.lower()
    dfv = pd.merge(dfv, dfp[['projectId', 'title']], how='left', left_on='projectId', right_on='projectId')
    
    #dfv['rawScore'] = 0
    #dfpp = load_passport_data()
    #if not dfpp.empty:
    #    dfpp['address'] = dfpp['address'].str.lower()
    #    dfv = pd.merge(dfv, dfpp[['address', 'rawScore']], how='left', left_on='voter', right_on='address')
    #del dfpp

    #df_ens = pd.read_csv('ens.csv')
    #df_ens['address'] = df_ens['address'].str.lower()
    
    #dfv = pd.merge(dfv, df_ens, how='left', left_on='voter', right_on='address')
    #dfv['voter_id'] = dfv['name'].fillna(dfv['voter'])
    dfv.drop_duplicates()
    st.session_state.dfv = dfv
    st.session_state.dfp = dfp
    st.session_state.round_data = round_data
    st.session_state.data_loaded = True

    return dfv, dfp, round_data

def get_time_left(target_time):
    now = datetime.now(timezone.utc)
    time_diff = target_time - now
    hours, remainder = divmod(time_diff.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if time_diff.days < 0:
        return f"0 days   0 hours   0 minutes"
    return f"{time_diff.days} days   {hours} hours   {minutes} minutes"


def get_beta_votes_data_online():
    # Set up headers with the session token
    QUESTION_ID = 7
    headers = {
        "Content-Type": "application/json",
        "X-Metabase-Session": 'fdfba7f0-c4ca-4fd3-ae92-a4a144f50d4f'
    }
    query_url = f"https://regendata.xyz/api/card/{QUESTION_ID}/query/json"
    response = requests.post(query_url, headers=headers)

    if response.status_code == 200:
        data = response.json()
    else:
        st.warning("Error:", response.status_code, response.text)
        data = []
    st.write(data)
    # Load in df
    df_votes = pd.DataFrame(data)

    df_votes['day_number'] = df_votes.groupby('round_id').cumcount() + 1
    # drop round_id and date
    df_votes.drop(['round_id', 'date'], axis=1, inplace=True)
    df_votes['program'] = "Beta"
    return df_votes

@st.cache_resource(ttl=9999999)
def get_beta_votes_data():
    df_votes = pd.read_csv('beta_by_day.csv')
    df_votes['day_number'] = df_votes.groupby('round_id').cumcount() + 1
    # drop round_id and date
    st.write(df_votes)
    df_votes.drop(['round_id', 'date'], axis=1, inplace=True)
    df_votes['program'] = "Beta"
    return df_votes

