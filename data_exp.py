import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

st.title('Uber pickups in NYC')

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
         'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = load_data(10000)
# Notify the reader that the data was successfully loaded.
data_load_state.text("Done! (using st.cache)")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

st.subheader('Number of pickups by hour')

bar_slot = st.empty()
hour_to_filter = st.slider('slide the hour bar to change the fig below', 0, 23, 17)

hist_values , bins = np.histogram(
    data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))

filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

hourly = pd.DataFrame({'bins':bins[:-1], 'hour':hist_values})
hourly.loc[hourly.bins == hour_to_filter, 'hour_concerned'] = hourly.hour
hourly.loc[hourly.bins == hour_to_filter , 'hour'] = 0
hourly.drop(columns = ['bins'], inplace=True)
hourly = hourly.fillna(0)
#st.write(hourly)
bar_slot.bar_chart(hourly, height=200)


st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=pdk.ViewState(
        latitude= 40.75,
        longitude= -74,
        zoom=11,
        pitch=60,
    ),
    layers=[
        pdk.Layer(
           'HexagonLayer',
           data=filtered_data,
           get_position='[lon, lat]',
           radius=200,
           elevation_scale=4,
           elevation_range=[0, 1000],
           pickable=True,
           extruded=True,
        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=filtered_data,
            get_position='[lon, lat]',
            get_color='[200, 30, 0, 160]',
            get_radius=200,
        ),
    ], 
))
