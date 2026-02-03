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
import time

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

# Helper function to get yfinance data with retry
def get_yfinance_data_with_retry(ticker_symbol, max_retries=3):
    """Get yfinance data with exponential backoff retry"""
    for attempt in range(max_retries):
        try:
            ticker_yf = yf.Ticker(ticker_symbol)
            expiry_dates = ticker_yf.options
            spot = ticker_yf.history(period="1d")['Close'].iloc[-1]
            return expiry_dates, spot, None
        except Exception as e:
            if "Rate limit" in str(e) or "Too Many Requests" in str(e):
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 2  # 2, 4, 8 seconds
                    time.sleep(wait_time)
                    continue
                else:
                    return None, None, f"Yahoo Finance rate limit exceeded. Please wait a few minutes and try again."
            else:
                return None, None, f"YFinance error: {str(e)}"
    return None, None, "Failed after retries"

# Fetch Barchart data
@st.cache_data(ttl=600)  # Increased cache time to 10 minutes
def fetch_barchart_options(ticker_symbol, expiry_offset=0):
    try:
        # Get expiration dates from yfinance with retry logic
        expiry_dates, spot, error = get_yfinance_data_with_retry(ticker_symbol)
        
        if error:
            return {'error': error, 'type': 'YFinanceRateLimitError', 'traceback': 'Yahoo Finance is rate limiting requests. Wait 2-5 minutes and refresh.'}
        
        if not expiry_dates:
            return {'error': f'No expiration dates found for {ticker_symbol}', 'type': 'NoOptionsError', 'traceback': 'Ticker has no options data available'}
        
        if expiry_offset >= len(expiry_dates):
            return {'error': f'Expiry offset {expiry_offset} too high, only {len(expiry_dates)} dates available', 'type': 'OffsetError', 'traceback': f'Available dates: {expiry_dates[:5]}'}
        
        expiry = expiry_dates[expiry_offset]
        
        geturl = f'https://www.barchart.com/etfs-funds/quotes/{ticker_symbol}/volatility-greeks'
        apiurl = 'https://www.barchart.com/proxies/core-api/v1/options/get'
        
        getheaders = {
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'accept-encoding': 'gzip, deflate, br',
            'accept-language': 'en-US,en;q=0.9',
            'cache-control': 'max-age=0',
            'upgrade-insecure-requests': '1',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.119 Safari/537.36'
        }
        
        getpay = {'page': 'all'}
        
        s = requests.Session()
        r = s.get(geturl, params=getpay, headers=getheaders, timeout=10)
        r.raise_for_status()
        
        headers = {
            'accept': 'application/json',
            'accept-encoding': 'gzip, deflate, br',
            'accept-language': 'en-US,en;q=0.9',
            'referer': f'https://www.barchart.com/etfs-funds/quotes/{ticker_symbol}/volatility-greeks',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.119 Safari/537.36',
            'x-xsrf-token': unquote(unquote(s.cookies.get_dict()['XSRF-TOKEN']))
        }
        
        payload = {
            'baseSymbol': ticker_symbol,
            'groupBy': 'optionType',
            'expirationDate': expiry,
            'meta': 'field.shortName,expirations,field.description',
            'orderBy': 'strikePrice',
            'orderDir': 'asc',
            'raw': '1',
            'fields': 'symbol,baseSymbol,strikePrice,lastPrice,volatility,delta,gamma,theta,vega,rho,volume,openInterest,optionType,daysToExpiration,expirationDate,tradeTime,averageVolatility,historicVolatility30d'
        }
        
        r = s.get(apiurl, params=payload, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        if 'data' not in data:
            return {'error': 'Invalid API response structure', 'type': 'APIResponseError', 'traceback': f'Response keys: {list(data.keys())}'}
        
        data_list = []
        for option_type, options in data['data'].items():
            for option in options:
                option['optionType'] = option_type
                data_list.append(option)
        
        if not data_list:
            return {'error': 'No options data returned from API', 'type': 'EmptyDataError', 'traceback': 'API returned empty options list'}
        
        df = pd.DataFrame(data_list)
        df['strikePrice'] = pd.to_numeric(df['strikePrice'], errors='coerce')
        df['openInterest'] = pd.to_numeric(df['openInterest'], errors='coerce').fillna(0).astype(int)
        df['gamma'] = pd.to_numeric(df['gamma'], errors='coerce').fillna(0)
        df['theta'] = pd.to_numeric(df['theta'], errors='coerce').fillna(0)
        
        calls = df[df['optionType'] == 'Call'].copy()
        puts = df[df['optionType'] == 'Put'].copy()
        
        return {'spot': spot, 'expiry': expiry, 'calls': calls, 'puts': puts}
    except Exception as e:
        import traceback
        error_details = {
            'error': str(e),
            'type': type(e).__name__,
            'traceback': traceback.format_exc()
        }
        return error_details

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

# Rate limit warning
st.info("⚠️ **Note:** If you see rate limit errors, wait 2-5 minutes before refreshing. Yahoo Finance limits requests from cloud IPs.")

# Controls
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
with col1:
    TICKER = st.text_input("Ticker", "SPY", key="ticker", label_visibility="collapsed").upper()
with col2:
    EXPIRY = st.number_input("Expiry", 0, 20, 1, key="expiry", label_visibility="collapsed")
with col3:
    RANGE_PCT = st.slider("Range", 5, 30, 15, key="range", label_visibility="collapsed")
with col4:
    if st.button("🔄"):
        st.cache_data.clear()
        st.rerun()

# Fetch data
result = fetch_barchart_options(TICKER, EXPIRY)

# Check for errors
if not result:
    st.error("❌ Failed to fetch data - returned None")
    st.stop()

if 'error' in result:
    if 'rate limit' in result['error'].lower():
        st.error("🚫 Yahoo Finance Rate Limit Reached")
        st.warning("""
        **What happened?** Yahoo Finance is limiting requests from Streamlit Cloud.
        
        **Solutions:**
        1. ⏰ Wait 2-5 minutes and click the refresh button (🔄)
        2. 🔄 Clear your browser cache and reload
        3. 💻 Run this app locally (no rate limits on your own IP)
        
        **Why does this happen?** Streamlit Cloud shares IP addresses, so Yahoo sees many requests from the same source.
        """)
    else:
        st.error(f"❌ Failed to fetch data")
        st.error(f"**Error Type:** {result['type']}")
        st.error(f"**Error Message:** {result['error']}")
    
    with st.expander("🔍 Technical Details"):
        st.code(result['traceback'])
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

# Get price data - fetch more historical data for full day view
@st.cache_data(ttl=300)
def get_prices(ticker):
    try:
        # Fetch 2 days of 5-min data to ensure we have full current day
        hist = yf.Ticker(ticker).history(period="2d", interval="5m")
        
        if hist.empty:
            return None
        
        # Filter to today's data only
        today = pd.Timestamp.now().normalize()
        hist_today = hist[hist.index.date >= today.date()]
        
        # If today's data is empty, use last available day
        if hist_today.empty:
            return hist
        
        return hist_today if not hist_today.empty else hist
    except:
        return None

prices = get_prices(TICKER)

# Show diagnostic metrics
st.caption(f"**{TICKER}** @ ${spot:.2f} | Expiry: {expiry}")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Strikes", len(gex_filt))
with col2:
    st.metric("Max Call GEX", f"{gex_filt['call_gex'].max():,.0f}")
with col3:
    st.metric("Max Put GEX", f"{gex_filt['put_gex'].max():,.0f}")
with col4:
    net = gex_filt['net_gex'].sum()
    st.metric("Net GEX", f"{net:,.0f}")
with col5:
    if prices is not None:
        st.metric("Price Bars", len(prices))
    else:
        st.metric("Price Bars", "N/A")

st.markdown("---")

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
    
    if prices is not None and len(prices) > 0:
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
        
        strikes = gex_filt['strike'].values
        call_gex_vals = gex_filt['call_gex'].values
        put_gex_vals = gex_filt['put_gex'].values
        total_gex = gex_filt['total_gex'].values
        
        # Normalize for color intensity
        max_gex = total_gex.max()
        if max_gex > 0:
            gex_norm = total_gex / max_gex
        else:
            gex_norm = np.zeros(len(total_gex))
        
        # Get FULL time range - this is critical!
        x_min = prices.index.min()
        x_max = prices.index.max()
        
        # TOP CHART: Gamma gradient zones - FULL WIDTH
        for i in range(len(strikes) - 1):
            intensity = gex_norm[i]
            
            # High gamma = GREEN (resistance), Low gamma = RED (support)
            if total_gex[i] > max_gex * 0.3:  # Top 30% = green
                opacity = min(0.7, intensity * 0.8)
                color = f'rgba(0, 255, 0, {opacity})'
            else:
                opacity = 0.3
                color = f'rgba(255, 0, 0, {opacity})'
            
            # CRITICAL: Add shape that spans FULL x-axis
            fig2.add_shape(
                type="rect",
                x0=x_min,
                x1=x_max,
                y0=strikes[i],
                y1=strikes[i + 1],
                fillcolor=color,
                layer='below',
                line_width=0,
                row=1, col=1
            )
        
        # BOTTOM CHART: Call vs Put dominance - FULL WIDTH
        for i in range(len(strikes) - 1):
            call_val = call_gex_vals[i] if i < len(call_gex_vals) else 0
            put_val = put_gex_vals[i] if i < len(put_gex_vals) else 0
            
            total = call_val + put_val
            if total > 0:
                if call_val > put_val:
                    # Call dominant = YELLOW/GOLD
                    ratio = call_val / total
                    opacity = min(0.6, ratio * 0.7)
                    color = f'rgba(255, 215, 0, {opacity})'
                else:
                    # Put dominant = BLUE
                    ratio = put_val / total
                    opacity = min(0.6, ratio * 0.7)
                    color = f'rgba(74, 158, 255, {opacity})'
            else:
                color = 'rgba(100, 100, 100, 0.1)'
            
            # CRITICAL: Full width shape
            fig2.add_shape(
                type="rect",
                x0=x_min,
                x1=x_max,
                y0=strikes[i],
                y1=strikes[i + 1],
                fillcolor=color,
                layer='below',
                line_width=0,
                row=2, col=1
            )
        
        # Add candlesticks AFTER gradients (on top)
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
                    increasing_fillcolor='rgba(0,255,0,0.3)',
                    decreasing_fillcolor='rgba(255,0,0,0.3)',
                    line=dict(width=1),
                    showlegend=False
                ),
                row=row, col=1
            )
            
            # Spot line
            fig2.add_hline(
                y=spot, 
                line=dict(color='#FFD700', width=2, dash='dot'),
                row=row, col=1
            )
        
        fig2.update_layout(
            height=850,
            plot_bgcolor='#0a0e27',
            paper_bgcolor='#0a0e27',
            font=dict(color='#fff', size=10),
            showlegend=False,
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=10, b=10),
            hovermode='x unified'
        )
        
        # Important: Set ranges to show full day
        fig2.update_xaxes(
            gridcolor='#1a1f3a',
            showgrid=True,
            range=[x_min, x_max],  # Show full time range
            fixedrange=False  # Allow zooming but start zoomed out
        )
        
        # Apply to both y-axes
        fig2.update_yaxes(
            gridcolor='#1a1f3a',
            tickformat='$.0f',
            range=[strikes.min() - 5, strikes.max() + 5],
            row=1, col=1
        )
        fig2.update_yaxes(
            gridcolor='#1a1f3a',
            tickformat='$.0f',
            range=[strikes.min() - 5, strikes.max() + 5],
            row=2, col=1
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("No price data available")

# Footer
st.caption(f"Data: Barchart + YFinance | Updated: {datetime.now().strftime('%H:%M:%S')}")
