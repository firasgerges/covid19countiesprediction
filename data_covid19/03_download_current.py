#!/usr/bin/env python

import pandas as pd

# Data Source:
# - note, requests.get() and pandas.read_html() will not be able to access
#   the entire table

url = 'https://www.nytimes.com/interactive/2020/us/coronavirus-us-cases.html?auth=login-google1tap&login=google1tap#g-cases-by-county'
datafile = 'Coronavirus in the U.S._ Latest Map and Case Count - The New York Times.htm'

dfs = pd.read_html(datafile)

print('----- 1. State-level data: COVID19 Cases & Deaths -----')
df_state_level = dfs[0]
print(dfs[0].head())

print('----- 2. How Virus Was Contracted - number of cases -----')
df_how = dfs[1]
print(dfs[1].head())

print('----- 3. County-level data -----')
df_county = dfs[2]
us_states = df_county['State'].unique()
print(f"US States: {'|'.join(us_states)}\n")

df_ny_county = df_county[df_county['State'] == 'New York']
ny_counties = df_ny_county['County'].unique()
print(f"NY Counties: {'|'.join(ny_counties)}\n")

print(df_ny_county)

df_ny_county.to_csv('ny_county_confirmed_cases_20200326_1400h_UTC.csv')
