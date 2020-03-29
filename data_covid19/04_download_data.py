#!/usr/bin/env python

import json
import requests

# https://corona.ndo.dev/api-docs/#/Timeseries/get_timespan

# ----- Example bash commands -----
# curl -X GET "https://data.corona-api.org/v1/daily?country=USA&state=NY" -H "accept: application/json" | jq
# curl -X GET "https://data.corona-api.org/v1/timespan?country=USA&time=week" -H "accept: application/json" | jq
# curl -X GET "https://data.corona-api.org/v1/timespan?country=USA&time=year" -H "accept: application/json" | jq

url = "https://data.corona-api.org/v1/timespan?country=USA&time=year"
headers = { "accept": "application/json" }

response = requests.get(url, headers=headers)
d = response.json()
ts = d['timeseries']
print(f"Length of timeseries: {len(ts)}")

with open('county-data.json', 'w') as f:
    f.write(json.dumps(d, indent=2))

