import streamlit as st
import pandas as pd
import numpy as np
import time

# Get some data.
data = np.random.randn(10, 3)

# Show the data as a chart.
chart = st.line_chart(data)
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(0,25):
    # Wait 1 second, so the change is clearer.
    for j in range(1,5):
        time.sleep(0.1)
        latest_iteration.text(f'Iteration {i*4+j}, progess {i*4+j}%')
        bar.progress(i*4 + j)
    # Grab some more data.
    data2 = np.random.randn(10, 1) + 5 * np.log(i+1)
    data3 = np.random.randn(10, 1) + i**1/4 - 5 * np.log(i+1)
    data4 = np.random.randn(10, 1) + 12 * np.log(i+1) - i**2/11
    #st.write(np.log(i))
    data2 = np.concatenate((data2,data3,data4), axis=1)
    #st.write(data2)
    # Append the new data to the existing chart.
    chart.add_rows(data2)