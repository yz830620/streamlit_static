import streamlit as st
import pandas as pd

st.title('My first app')
st.write("Here's our first attempt at using data to create a table:")
st.write(pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
}))
st.write(
    """
# Apps with widgets!
"""
)
x = st.slider("Select a number", 0, 100)
st.write("You selected", x)
