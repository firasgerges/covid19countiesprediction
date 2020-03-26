#!/usr/bin/env python

import argparse
import os
import pandas as pd
import re


DATAFILE = 'time_series_19-covid-Confirmed_archived_0325.csv'
DATAFILE_PREFIX, _ = os.path.splitext(DATAFILE)
COUNTY_FILE = DATAFILE_PREFIX + '_uscounties.csv'
NY_FILE = DATAFILE_PREFIX + '_nycounties.csv'

# ----- Worldwide data -----
df_worldwide = pd.read_csv(DATAFILE)
countries = df_worldwide['Country/Region'].unique()

# ----- US data -----
df_us = df_worldwide.loc[df_worldwide['Country/Region'] == 'US']
us_states = df_us['Province/State'].unique()
us_counties = [ x for x in us_states if 'County' in x ]
df_counties = df_us[df_us['Province/State'].isin(us_counties)]

# ----- NY data -----
ny_counties = [ x for x in us_counties if ', NY' in x ]
df_ny = df_counties[df_counties['Province/State'].isin(ny_counties)]


def show_counties():
    print(f"US Counties:\n{'|'.join(sorted(us_counties))}\n")
    print(f"NY Counties:\n{'|'.join(sorted(ny_counties))}\n")


def explore_data():
    print(f"Countries:\n{'|'.join(sorted(countries))}\n")
    print(f"US States:\n{'|'.join(sorted(us_states))}\n")
    show_counties()
    
    
if __name__ == "__main__":
    show_counties()
    print('----- County Level Data -----')
    print(df_counties.head())
    df_counties.to_csv(COUNTY_FILE)
    print(f'\nCreated: {COUNTY_FILE}\n')

    print('----- NY County Level Data -----')
    print(df_ny.head())
    df_ny.to_csv(NY_FILE)
    print(f'\nCreated: {NY_FILE}\n')
