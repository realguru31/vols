import streamlit as st
from tvDatafeed import TvDatafeed, Interval

# Initialize the feed (No-login method)
tv = TvDatafeed()

st.title("SPY 1-Minute Data")

# Fetch 1-minute data for SPY
# Symbol: SPY, Exchange: AMEX, Interval: 1 minute
data = tv.get_hist(symbol='SPY', exchange='AMEX', interval=Interval.in_1_minute, n_bars=100)

if data is not None:
    st.subheader("Last 100 Minutes of SPY")
    # Display line chart of closing prices
    st.line_chart(data['close'])
    # Display raw dataframe
    st.write(data)
else:
    st.error("Failed to fetch data. Check your connection or symbol/exchange.")
