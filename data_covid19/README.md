## README - COVID19 Data

### 1. Extract-Transform-Load (ETL) for COVID19 Data

#### 1.1 Download the data

- **Data Source**: [time_series_19-covid-Confirmed_archived_0325.csv](https://github.com/CSSEGISandData/COVID-19/blob/master/archived_data/archived_time_series/time_series_19-covid-Confirmed_archived_0325.csv) and [daily reports](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_daily_reports)

```bash
bash 01_download_data.sh
```

#### 1.2 Clean the data

```bash
python3 02_clean_data.py
```

#### 1.3 Download data source for current county-by-county data (by NY Times)

- **Data Source**: [https://www.nytimes.com/interactive/2020/us/coronavirus-us-cases.html](https://www.nytimes.com/interactive/2020/us/coronavirus-us-cases.html?auth=login-google1tap&login=google1tap#g-cases-by-county)

