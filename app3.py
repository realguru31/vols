import streamlit as st
import requests
import pandas as pd
from urllib.parse import unquote
from datetime import datetime
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.interpolate import interp1d

# Page config
st.set_page_config(page_title="GEX Analyzer - VolSignals Style", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #0a0e27;}
    .stApp {background-color: #0a0e27;}
    h1, h2, h3, h4 {color: #ffffff !important;}
    .stMarkdown {color: #ffffff;}
</style>
""", unsafe_allow_html=True)

# Fetch Barchart data
@st.cache_data(ttl=300)
def fetch_barchart_options(ticker_symbol, expiry_offset=0):
    try:
        ticker_yf = yf.Ticker(ticker_symbol)
        expiry_dates = ticker_yf.options
        
        if not expiry_dates or expiry_offset >= len(expiry_dates):
            return None
        
        expiry = expiry_dates[expiry_offset]
        spot = ticker_yf.history(period="1d")['Close'].iloc[-1]
        
        geturl = f'https://www.barchart.com/etfs-funds/quotes/{ticker_symbol}/volatility-greeks'
        apiurl = 'https://www.barchart.com/proxies/core-api/v1/options/get'
        
        getheaders = {
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        s = requests.Session()
        r = s.get(geturl, headers=getheaders, timeout=10)
        r.raise_for_status()
        
        headers = {
            'accept': 'application/json',
            'referer': geturl,
            'user-agent': 'Mozilla/5.0',
            'x-xsrf-token': unquote(unquote(s.cookies.get_dict()['XSRF-TOKEN']))
        }
        
        payload = {
            'baseSymbol': ticker_symbol,
            'groupBy': 'optionType',
            'expirationDate': expiry,
            'orderBy': 'strikePrice',
            'orderDir': 'asc',
            'raw': '1',
            'fields': 'strikePrice,openInterest,gamma,theta,optionType'
        }
        
        r = s.get(apiurl, params=payload, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        data_list = []
        for option_type, options in data['data'].items():
            for option in options:
                option['optionType'] = option_type
                data_list.append(option)
        
        df = pd.DataFrame(data_list)
        df['strikePrice'] = pd.to_numeric(df['strikePrice'], errors='coerce')
        df['openInterest'] = pd.to_numeric(df['openInterest'], errors='coerce').fillna(0).astype(int)
        df['gamma'] = pd.to_numeric(df['gamma'], errors='coerce').fillna(0)
        df['theta'] = pd.to_numeric(df['theta'], errors='coerce').fillna(0)
        
        calls = df[df['optionType'] == 'Call'].copy()
        puts = df[df['optionType'] == 'Put'].copy()
        
        return {'spot': spot, 'expiry': expiry, 'calls': calls, 'puts': puts}
    except:
        return None

# Compute GEX
def compute_gex(calls, puts):
    all_strikes = sorted(list(set(calls['strikePrice'].tolist() + puts['strikePrice'].tolist())))
    gex_data = []
    
    for K in all_strikes:
        if pd.isna(K):
            continue
        
        call_gex = put_gex = 0
        
        call_data = calls[calls['strikePrice'] == K]
        if not call_data.empty:
            row = call_data.iloc[0]
            oi = int(row['openInterest'])
            gamma = float(row['gamma'])
            if oi > 0 and gamma > 0:
                call_gex = gamma * oi * 100
        
        put_data = puts[puts['strikePrice'] == K]
        if not put_data.empty:
            row = put_data.iloc[0]
            oi = int(row['openInterest'])
            gamma = float(row['gamma'])
            if oi > 0 and gamma > 0:
                put_gex = gamma * oi * 100
        
        gex_data.append({
            'strike': K,
            'call_gex': call_gex,
            'put_gex': put_gex,
            'net_gex': call_gex - put_gex,
            'total_gex': call_gex + put_gex
        })
    
    return pd.DataFrame(gex_data)

# Title
st.markdown("## 📊 GEX Profile Analyzer")

# Controls
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
with col1:
    TICKER = st.text_input("", "SPY", key="ticker").upper()
with col2:
    EXPIRY = st.number_input("", 0, 20, 1, key="expiry")
with col3:
    RANGE_PCT = st.slider("", 5, 30, 15, key="range")
with col4:
    if st.button("🔄"):
        st.cache_data.clear()
        st.rerun()

# Fetch data
result = fetch_barchart_options(TICKER, EXPIRY)
if not result:
    st.error("Failed to fetch data")
    st.stop()

spot = result['spot']
expiry = result['expiry']
gex_df = compute_gex(result['calls'], result['puts'])

# Filter
price_range = spot * (RANGE_PCT / 100)
gex_filt = gex_df[
    (gex_df['strike'] >= spot - price_range) &
    (gex_df['strike'] <= spot + price_range) &
    ((gex_df['call_gex'] != 0) | (gex_df['put_gex'] != 0))
]

if len(gex_filt) == 0:
    st.warning("No data")
    st.stop()

# Get price data
@st.cache_data(ttl=300)
def get_prices(ticker):
    try:
        hist = yf.Ticker(ticker).history(period="1d", interval="5m")
        return hist if not hist.empty else None
    except:
        return None

prices = get_prices(TICKER)

# Layout
col_left, col_right = st.columns([1, 2])

with col_left:
    st.markdown("#### Positions by Strike")
    
    fig1 = go.Figure()
    
    # Blue bars (puts) pointing RIGHT
    fig1.add_trace(go.Bar(
        y=gex_filt['strike'],
        x=gex_filt['put_gex'],
        orientation='h',
        marker=dict(color='#4A9EFF'),
        name='Put',
        hovertemplate='$%{y:.0f}<br>%{x:,.0f}<extra></extra>'
    ))
    
    # Orange bars (calls) pointing LEFT
    fig1.add_trace(go.Bar(
        y=gex_filt['strike'],
        x=-gex_filt['call_gex'],
        orientation='h',
        marker=dict(color='#FFB84D'),
        name='Call',
        hovertemplate='$%{y:.0f}<br>%{x:,.0f}<extra></extra>'
    ))
    
    # Yellow spot line
    fig1.add_hline(y=spot, line=dict(color='#FFD700', width=3))
    
    fig1.update_layout(
        barmode='overlay',
        height=850,
        plot_bgcolor='#0a0e27',
        paper_bgcolor='#0a0e27',
        font=dict(color='#fff', size=10),
        xaxis=dict(gridcolor='#1a1f3a', tickformat=','),
        yaxis=dict(gridcolor='#1a1f3a', tickformat='$.0f'),
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    
    st.plotly_chart(fig1, use_container_width=True)

with col_right:
    st.markdown("#### Gamma & Charm Gradients")
    
    if prices is not None:
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
        
        strikes = gex_filt['strike'].values
        total_gex = gex_filt['total_gex'].values
        
        # Normalize
        if len(total_gex) > 0 and total_gex.max() > 0:
            gex_norm = total_gex / total_gex.max()
        else:
            gex_norm = np.zeros(len(total_gex))
        
        # TOP: Gamma gradient (green high, red low)
        for i in range(len(strikes) - 1):
            intensity = gex_norm[i]
            if intensity > 0.5:
                color = f'rgba(0, 255, 0, {intensity * 0.4})'
            else:
                color = f'rgba(255, 0, 0, {(1 - intensity) * 0.4})'
            
            fig2.add_hrect(y0=strikes[i], y1=strikes[i + 1],
                          fillcolor=color, line_width=0, row=1, col=1)
        
        # BOTTOM: Charm gradient (yellow above, blue below)
        for i in range(len(strikes) - 1):
            if strikes[i] > spot:
                color = 'rgba(255, 215, 0, 0.25)'
            else:
                color = 'rgba(74, 158, 255, 0.25)'
            
            fig2.add_hrect(y0=strikes[i], y1=strikes[i + 1],
                          fillcolor=color, line_width=0, row=2, col=1)
        
        # Candlesticks on both
        for row in [1, 2]:
            fig2.add_trace(
                go.Candlestick(
                    x=prices.index,
                    open=prices['Open'],
                    high=prices['High'],
                    low=prices['Low'],
                    close=prices['Close'],
                    increasing_line_color='#00FF00',
                    decreasing_line_color='#FF0000',
                    line=dict(width=1)
                ),
                row=row, col=1
            )
            
            # Spot line
            fig2.add_hline(y=spot, line=dict(color='#FFD700', width=2, dash='dot'), row=row, col=1)
        
        fig2.update_layout(
            height=850,
            plot_bgcolor='#0a0e27',
            paper_bgcolor='#0a0e27',
            font=dict(color='#fff', size=10),
            showlegend=False,
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        
        fig2.update_xaxes(gridcolor='#1a1f3a')
        fig2.update_yaxes(gridcolor='#1a1f3a', tickformat='$.0f')
        
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("No price data")

# Footer
st.caption(f"{TICKER} @ ${spot:.2f} | {expiry} | Barchart + YFinance | {datetime.now().strftime('%H:%M')}")
