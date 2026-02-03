import streamlit as st
import requests
import pandas as pd
from urllib.parse import unquote
from datetime import datetime, timedelta
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
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
                    wait_time = (2 ** attempt) * 2
                    time.sleep(wait_time)
                    continue
                else:
                    return None, None, f"Yahoo Finance rate limit exceeded. Please wait a few minutes and try again."
            else:
                return None, None, f"YFinance error: {str(e)}"
    return None, None, "Failed after retries"

# Fetch Barchart data
@st.cache_data(ttl=600)
def fetch_barchart_options(ticker_symbol, expiry_offset=0):
    try:
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
        df['delta'] = pd.to_numeric(df['delta'], errors='coerce').fillna(0)
        df['vega'] = pd.to_numeric(df['vega'], errors='coerce').fillna(0)
        df['volatility'] = pd.to_numeric(df['volatility'], errors='coerce').fillna(0)
        
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
        
        call_gex = 0
        put_gex = 0
        
        # Process calls
        call_data = calls[calls['strikePrice'] == K]
        if not call_data.empty:
            row = call_data.iloc[0]
            oi = int(row['openInterest'])
            gamma = float(row['gamma'])
            
            if oi > 0 and gamma > 0:
                call_gex = gamma * oi * 100
        
        # Process puts
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
            'total_gex': call_gex + put_gex,
            'sell_gamma': -call_gex if call_gex > 0 else 0,
            'buy_gamma': put_gex if put_gex > 0 else 0,
        })
    
    return pd.DataFrame(gex_data)

# Title
st.markdown("## 📊 GEX Profile Analyzer - Professional Gradient View")

# Rate limit warning
st.info("⚠️ **Note:** If you see rate limit errors, wait 2-5 minutes before refreshing.")

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

# Show loading state
with st.spinner("Fetching data..."):
    result = fetch_barchart_options(TICKER, EXPIRY)

# Check for errors
if 'error' in result:
    st.error(f"❌ {result['error']}")
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
    st.warning("No GEX data in selected range.")
    st.stop()

# Get price data
@st.cache_data(ttl=300)
def get_prices(ticker):
    try:
        hist = yf.Ticker(ticker).history(period="2d", interval="5m")
        if hist.empty:
            return None
        today = pd.Timestamp.now().normalize()
        hist_today = hist[hist.index.date >= today.date()]
        return hist_today if not hist_today.empty else hist
    except:
        return None

with st.spinner("Fetching price data..."):
    prices = get_prices(TICKER)

# Show metrics
st.caption(f"**{TICKER}** @ ${spot:.2f} | Expiry: {expiry}")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Strikes", len(gex_filt))
with col2:
    st.metric("Max Call GEX", f"{gex_filt['call_gex'].max():,.0f}")
with col3:
    st.metric("Max Put GEX", f"{gex_filt['put_gex'].max():,.0f}")
with col4:
    net = gex_filt['net_gex'].sum()
    st.metric("Net GEX", f"{net:,.0f}")

st.markdown("---")

# Create smooth gradient heatmaps
st.markdown("#### 🌊 GEX Heatmaps (Professional Gradient View)")

if prices is not None and len(prices) > 0:
    # Prepare data
    strikes = gex_filt['strike'].values
    total_gex = gex_filt['total_gex'].values
    net_gex = gex_filt['net_gex'].values
    
    # Create time grid from price data
    time_values = prices.index.astype(np.int64) // 10**9
    time_grid = np.linspace(time_values.min(), time_values.max(), 100)
    
    # Create strike grid
    strike_grid = np.linspace(strikes.min(), strikes.max(), 100)
    
    # Create 2D grid
    time_mesh, strike_mesh = np.meshgrid(time_grid, strike_grid)
    
    # Prepare data for interpolation - simpler approach
    grid_total = np.zeros((len(strike_grid), len(time_grid)))
    grid_net = np.zeros((len(strike_grid), len(time_grid)))
    
    # For each strike, spread the gamma value across time dimension
    for i, strike in enumerate(strikes):
        if i < len(total_gex):
            # Find closest strike in grid
            idx = np.argmin(np.abs(strike_grid - strike))
            grid_total[idx, :] = total_gex[i]
            grid_net[idx, :] = net_gex[i]
    
    # Apply Gaussian smoothing for smooth gradients
    grid_total = gaussian_filter(grid_total, sigma=2)
    grid_net = gaussian_filter(grid_net, sigma=2)
    
    # Convert time grid back to datetime
    time_dates = pd.to_datetime(time_grid, unit='s')
    
    # Create the professional gradient chart
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Gamma Exposure Heatmap", "Net GEX Heatmap"),
        row_heights=[0.5, 0.5]
    )
    
    # TOP: Total Gamma Heatmap
    # Normalize data for colorscale
    max_gamma = np.max(grid_total) if np.max(grid_total) > 0 else 1
    grid_total_normalized = grid_total / max_gamma
    
    fig.add_trace(
        go.Heatmap(
            x=time_dates,
            y=strike_grid,
            z=grid_total_normalized,
            colorscale=[
                [0, 'rgba(255, 0, 0, 0.1)'],
                [0.3, 'rgba(255, 100, 0, 0.4)'],
                [0.6, 'rgba(255, 255, 0, 0.7)'],
                [1, 'rgba(0, 255, 0, 0.9)']
            ],
            showscale=True,
            colorbar=dict(
                title=dict(
                    text="Gamma Density",
                    font=dict(color='white')
                ),
                tickfont=dict(color='white'),
                tickformat=".1%"
            ),
            hovertemplate="Time: %{x}<br>Strike: $%{y:.0f}<br>Gamma: %{z:,.0f}<extra></extra>",
            name="Gamma Exposure"
        ),
        row=1, col=1
    )
    
    # BOTTOM: Net GEX Heatmap
    # Normalize net GEX for colorscale (0-1 range)
    max_abs_net = max(abs(np.min(grid_net)), abs(np.max(grid_net))) if grid_net.size > 0 else 1
    if max_abs_net > 0:
        grid_net_normalized = grid_net / (max_abs_net * 2) + 0.5  # Scale to 0-1 range
    else:
        grid_net_normalized = np.ones_like(grid_net) * 0.5
    
    fig.add_trace(
        go.Heatmap(
            x=time_dates,
            y=strike_grid,
            z=grid_net_normalized,
            colorscale=[
                [0, 'rgba(0, 100, 255, 0.9)'],      # Strong puts (blue)
                [0.25, 'rgba(100, 150, 255, 0.6)'], # Moderate puts
                [0.5, 'rgba(150, 150, 150, 0.3)'],  # Balanced
                [0.75, 'rgba(255, 200, 100, 0.6)'], # Moderate calls
                [1, 'rgba(255, 215, 0, 0.9)']       # Strong calls (gold)
            ],
            showscale=True,
            colorbar=dict(
                title=dict(
                    text="Net GEX<br>← Puts | Calls →",
                    font=dict(color='white')
                ),
                tickfont=dict(color='white'),
                tickvals=[0, 0.25, 0.5, 0.75, 1],
                ticktext=["Strong Puts", "Moderate Puts", "Balanced", "Moderate Calls", "Strong Calls"]
            ),
            hovertemplate="Time: %{x}<br>Strike: $%{y:.0f}<br>Net GEX: %{z:,.0f}<extra></extra>",
            name="Net GEX"
        ),
        row=2, col=1
    )
    
    # Add price candlesticks
    for row in [1, 2]:
        fig.add_trace(
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
                showlegend=False,
                name="Price"
            ),
            row=row, col=1
        )
        
        # Add spot price line
        fig.add_hline(
            y=spot,
            line=dict(color='#FFFFFF', width=2, dash='solid'),
            opacity=0.8,
            row=row, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=900,
        plot_bgcolor='#0a0e27',
        paper_bgcolor='#0a0e27',
        font=dict(color='white', size=12),
        showlegend=False,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode='x unified'
    )
    
    # Update axes
    for row in [1, 2]:
        fig.update_xaxes(
            gridcolor='rgba(255,255,255,0.1)',
            showgrid=True,
            tickformat="%H:%M",
            row=row, col=1
        )
        
        fig.update_yaxes(
            gridcolor='rgba(255,255,255,0.1)',
            showgrid=True,
            tickformat='$.0f',
            title="Strike Price",
            row=row, col=1
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add explanation
    with st.expander("📊 Chart Interpretation Guide", expanded=True):
        st.markdown("""
        ### Gamma Exposure Heatmap (Top Chart)
        - **🟢 Bright Green Areas**: High gamma concentration
        - **🟡 Yellow Areas**: Medium gamma exposure
        - **🔴 Red Areas**: Low gamma exposure
        
        ### Net GEX Heatmap (Bottom Chart)  
        - **🟡 Gold/Yellow Areas**: Call gamma dominant (resistance)
        - **🔵 Blue Areas**: Put gamma dominant (support)
        - **⚪ Gray Areas**: Balanced gamma
        
        ### How to Use This:
        1. **Support/Resistance**: Dense green areas indicate strong gamma
        2. **Gamma Walls**: Vertical bands show concentrated gamma
        3. **Price Magnet Areas**: High gamma areas often act as price magnets
        """)
    
else:
    st.warning("No intraday price data available for heatmap visualization.")
    
    # Simple 1D view
    st.markdown("#### 📊 Gamma Distribution")
    
    fig_fallback = go.Figure()
    
    # Add smooth area plot
    fig_fallback.add_trace(go.Scatter(
        x=gex_filt['strike'],
        y=gex_filt['total_gex'],
        fill='tozeroy',
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(0, 255, 0, 0.3)',
        name='Total Gamma'
    ))
    
    fig_fallback.add_trace(go.Scatter(
        x=gex_filt['strike'],
        y=gex_filt['total_gex'],
        mode='lines',
        line=dict(color='green', width=2),
        name='Total Gamma'
    ))
    
    fig_fallback.add_trace(go.Scatter(
        x=gex_filt['strike'],
        y=gex_filt['net_gex'],
        mode='lines',
        line=dict(color='gold', width=3),
        name='Net GEX'
    ))
    
    fig_fallback.add_vline(
        x=spot,
        line=dict(color='white', width=2, dash='dash'),
        annotation_text=f'Spot: ${spot:.2f}'
    )
    
    fig_fallback.update_layout(
        height=500,
        plot_bgcolor='#0a0e27',
        paper_bgcolor='#0a0e27',
        font=dict(color='white', size=12),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            tickformat='$.0f',
            title='Strike Price'
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            tickformat=',',
            title='Gamma Exposure'
        )
    )
    
    st.plotly_chart(fig_fallback, use_container_width=True)

# Original bar chart view
st.markdown("---")
st.markdown("#### 📊 Positions by Strike")

fig_bars = go.Figure()

fig_bars.add_trace(go.Bar(
    y=gex_filt['strike'],
    x=gex_filt['put_gex'],
    orientation='h',
    marker=dict(color='#4A9EFF'),
    name='Put'
))

fig_bars.add_trace(go.Bar(
    y=gex_filt['strike'],
    x=-gex_filt['call_gex'],
    orientation='h',
    marker=dict(color='#FFB84D'),
    name='Call'
))

fig_bars.add_hline(y=spot, line=dict(color='#FFD700', width=3))

fig_bars.update_layout(
    barmode='overlay',
    height=400,
    plot_bgcolor='#0a0e27',
    paper_bgcolor='#0a0e27',
    font=dict(color='white', size=10),
    xaxis=dict(gridcolor='#1a1f3a', tickformat=','),
    yaxis=dict(gridcolor='#1a1f3a', tickformat='$.0f'),
    showlegend=False
)

st.plotly_chart(fig_bars, use_container_width=True)

# Footer
st.markdown("---")
if prices is not None and len(prices) > 0:
    time_range = f"{prices.index.min().strftime('%H:%M')} - {prices.index.max().strftime('%H:%M')}"
    st.caption(f"Data: Barchart + YFinance | {time_range} | {datetime.now().strftime('%H:%M:%S')}")
else:
    st.caption(f"Data: Barchart + YFinance | {datetime.now().strftime('%H:%M:%S')}")
