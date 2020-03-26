#!/usr/bin/env python

import argparse
import os
import pandas as pd
import re


DATA_FILE = 'time_series_19-covid-Confirmed_archived_0325.csv'
COUNTY_FILE, _ = os.path.splitext(DATA_FILE)
COUNTY_FILE += '_uscounties.csv'


df_worldwide = pd.read_csv(DATA_FILE)
df_us = df_worldwide.loc[df_worldwide['Country/Region'] == 'US']
us_states = df_us['Province/State'].unique()
us_counties = [ x for x in us_states if 'County' in x ]
df_counties = df_us[df_us['Province/State'].isin(us_counties)]


def explore_data():
    countries = df_worldwide['Country/Region'].unique()
    print(f"Countries:\n{'|'.join(sorted(countries))}\n")
    print(f"US States:\n{'|'.join(sorted(us_states))}\n")
    print(f"US Counties:\n{'|'.join(sorted(us_counties))}")
    
    
if __name__ == "__main__":
    print('----- County Level Data -----')
    print(df_counties.head())
    df_counties.to_csv(COUNTY_FILE)
    print(f'\nCreated: {COUNTY_FILE}')
