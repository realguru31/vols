import streamlit as st
import pandas as pd
from tvDatafeed import TvDatafeed, Interval

st.title("Data Dashboard")

# Initialize the feed
tv = TvDatafeed()

# Define the asset you want to view
symbol = 'XAUUSD' # Gold
exchange = 'OANDA'
n_bars = 100

st.subheader(f"{symbol} 1-Minute Data")

# Fetch data
data = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_1_minute, n_bars=n_bars)

if data is not None and not data.empty:
    # Ensure the dataframe is correctly formatted for charting
    st.line_chart(data['close'])
    st.write(f"Showing last {n_bars} rows of data:")
    st.dataframe(data)
else:
    st.error(f"Failed to fetch data for {symbol} on {exchange}.")
    # Optional: Display a blank chart with the expected range if you want to see the axis scale
    # st.line_chart(pd.DataFrame({'close': [5000, 5100]})) 
