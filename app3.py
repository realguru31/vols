import streamlit as st
import requests
import pandas as pd
from urllib.parse import unquote
from datetime import datetime, timedelta
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from math import log, sqrt, exp
from scipy.stats import norm
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

# -----------------------------
# Black-Scholes Greeks Functions
# -----------------------------
def bs_greeks(S, K, T, r, sigma, option_type="call"):
    """Return delta, gamma, and charm for option."""
    if T <= 0 or sigma <= 0:
        return 0, 0, 0
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    
    delta = norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * sqrt(T))
    
    # Charm Calculation
    sqrt_T = sqrt(T)
    N_prime_d1 = norm.pdf(d1)
    charm = -exp(-r * T) * N_prime_d1 * ((r / (sigma * sqrt_T)) - (d1 / (2 * T)))
    
    return delta, gamma, charm

def estimate_iv_from_historical(ticker_obj, days=30):
    """Calculate historical volatility as IV fallback"""
    try:
        hist = ticker_obj.history(period=f"{days}d")
        if len(hist) < 2:
            return None
        returns = np.log(hist['Close'] / hist['Close'].shift(1))
        historical_vol = returns.std() * np.sqrt(252)
        return historical_vol
    except:
        return None

def get_vix_as_iv():
    """Use VIX as SPY IV estimate"""
    try:
        vix = yf.Ticker("^VIX")
        vix_value = vix.history(period="1d")['Close'].iloc[-1]
        iv_estimate = vix_value / 100
        return iv_estimate
    except:
        return None

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
        df['volatility'] = pd.to_numeric(df['volatility'], errors='coerce').fillna(0)
        
        calls = df[df['optionType'] == 'Call'].copy()
        puts = df[df['optionType'] == 'Put'].copy()
        
        # Calculate days to expiration
        expiry_date = datetime.strptime(expiry, "%Y-%m-%d")
        days_to_expiry = (expiry_date - datetime.now()).days
        T = max(days_to_expiry / 365, 1/252)  # Minimum 1 trading day
        
        return {'spot': spot, 'expiry': expiry, 'T': T, 'calls': calls, 'puts': puts}
    except Exception as e:
        import traceback
        error_details = {
            'error': str(e),
            'type': type(e).__name__,
            'traceback': traceback.format_exc()
        }
        return error_details

# Compute GEX with Black-Scholes fallback
def compute_gex_with_bs(calls, puts, spot, T, r=0.05):
    all_strikes = sorted(list(set(calls['strikePrice'].tolist() + puts['strikePrice'].tolist())))
    gex_data = []
    
    # Check IV quality
    calls_with_iv = calls[calls['volatility'] > 0]
    puts_with_iv = puts[puts['volatility'] > 0]
    iv_coverage = (len(calls_with_iv) + len(puts_with_iv)) / (len(calls) + len(puts)) * 100
    
    # Choose fallback IV if needed
    fallback_iv = None
    if iv_coverage < 50:
        # Try VIX for SPY, otherwise use historical
        if "SPY" in calls['baseSymbol'].iloc[0] if not calls.empty else False:
            fallback_iv = get_vix_as_iv()
        if fallback_iv is None:
            ticker_yf = yf.Ticker(calls['baseSymbol'].iloc[0] if not calls.empty else "SPY")
            fallback_iv = estimate_iv_from_historical(ticker_yf, days=30)
        if fallback_iv is None:
            fallback_iv = 0.15
    
    for K in all_strikes:
        if pd.isna(K):
            continue
        
        call_gex = 0
        put_gex = 0
        call_charm = 0
        put_charm = 0
        
        # Process calls
        call_data = calls[calls['strikePrice'] == K]
        if not call_data.empty:
            row = call_data.iloc[0]
            oi = int(row['openInterest'])
            iv = float(row['volatility'])
            
            if iv <= 0 and fallback_iv is not None:
                iv = fallback_iv
            
            if oi > 0 and iv > 0:
                _, gamma, charm = bs_greeks(spot, K, T, r, iv, "call")
                call_gex = gamma * oi * 100
                call_charm = charm * oi * 100
        
        # Process puts
        put_data = puts[puts['strikePrice'] == K]
        if not put_data.empty:
            row = put_data.iloc[0]
            oi = int(row['openInterest'])
            iv = float(row['volatility'])
            
            if iv <= 0 and fallback_iv is not None:
                iv = fallback_iv
            
            if oi > 0 and iv > 0:
                _, gamma, charm = bs_greeks(spot, K, T, r, iv, "put")
                put_gex = gamma * oi * 100
                put_charm = charm * oi * 100
        
        gex_data.append({
            'strike': K,
            'call_gex': call_gex,
            'put_gex': put_gex,
            'net_gex': call_gex - put_gex,
            'total_gex': call_gex + put_gex,
            'sell_gamma': -call_gex if call_gex > 0 else 0,
            'buy_gamma': put_gex if put_gex > 0 else 0,
            'call_charm': call_charm,
            'put_charm': put_charm,
            'net_charm': call_charm + put_charm
        })
    
    return pd.DataFrame(gex_data), fallback_iv

# Title
st.markdown("## 📊 GEX Profile Analyzer with Curve Plots")

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
T = result['T']
gex_df, fallback_iv = compute_gex_with_bs(result['calls'], result['puts'], spot, T)

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
        hist = yf.Ticker(ticker).history(period="2d", interval="5m")
        
        if hist.empty:
            return None
        
        today = pd.Timestamp.now().normalize()
        hist_today = hist[hist.index.date >= today.date()]
        
        return hist_today if not hist_today.empty else hist
    except:
        return None

prices = get_prices(TICKER)

# Show diagnostic metrics
st.caption(f"**{TICKER}** @ ${spot:.2f} | Expiry: {expiry} | T={T:.3f}y")
if fallback_iv:
    st.caption(f"⚠️ Using estimated IV: {fallback_iv:.4f}")

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
    total_charm = gex_filt['net_charm'].sum()
    st.metric("Net Charm", f"{total_charm:,.0f}")

st.markdown("---")

# NEW: Curve Plot Section
st.markdown("#### 📈 GEX Density Curves (Smooth Plots)")

# Prepare data for smooth curves
strikes = gex_filt['strike'].values
sell_gamma_plot = np.abs(gex_filt['sell_gamma'].values)
buy_gamma_plot = np.abs(gex_filt['buy_gamma'].values)
net_gex_plot = np.abs(gex_filt['net_gex'].values)
total_abs_gamma_plot = sell_gamma_plot + buy_gamma_plot

# Create smooth curves using Gaussian filter
if len(strikes) > 3:
    sigma = 1.5
    sell_gamma_smooth = gaussian_filter1d(sell_gamma_plot, sigma=sigma)
    buy_gamma_smooth = gaussian_filter1d(buy_gamma_plot, sigma=sigma)
    net_gex_smooth = gaussian_filter1d(net_gex_plot, sigma=sigma)
    total_abs_gamma_smooth = gaussian_filter1d(total_abs_gamma_plot, sigma=sigma)
else:
    sell_gamma_smooth = sell_gamma_plot
    buy_gamma_smooth = buy_gamma_plot
    net_gex_smooth = net_gex_plot
    total_abs_gamma_smooth = total_abs_gamma_plot

# Create curve plot
curve_fig = go.Figure()

# Add smooth curves
curve_fig.add_trace(go.Scatter(
    x=strikes,
    y=sell_gamma_smooth,
    mode='lines',
    name='Sell Gamma (|values|)',
    line=dict(color='green', width=2.5),
    opacity=0.8
))

curve_fig.add_trace(go.Scatter(
    x=strikes,
    y=buy_gamma_smooth,
    mode='lines',
    name='Buy Gamma (|values|)',
    line=dict(color='red', width=2.5),
    opacity=0.8
))

curve_fig.add_trace(go.Scatter(
    x=strikes,
    y=net_gex_smooth,
    mode='lines',
    name='Net GEX (|values|)',
    line=dict(color='gold', width=3),
    opacity=0.9
))

curve_fig.add_trace(go.Scatter(
    x=strikes,
    y=total_abs_gamma_smooth,
    mode='lines',
    name='Total Abs Gamma (|values|)',
    line=dict(color='orange', width=2.5),
    opacity=0.8
))

# Add spot line
curve_fig.add_vline(
    x=spot,
    line=dict(color='blue', width=2, dash='dash'),
    opacity=0.8,
    annotation_text=f'Spot: ${spot:.2f}',
    annotation_position="top right"
)

# Update layout
curve_fig.update_layout(
    height=400,
    plot_bgcolor='#0a0e27',
    paper_bgcolor='#0a0e27',
    font=dict(color='#fff', size=10),
    xaxis=dict(
        gridcolor='#1a1f3a',
        tickformat='$.0f',
        title='Strike Price'
    ),
    yaxis=dict(
        gridcolor='#1a1f3a',
        tickformat=',',
        title='Gamma Density (Absolute Values)',
        rangemode='tozero'
    ),
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor='rgba(10, 14, 39, 0.8)',
        bordercolor='#1a1f3a',
        borderwidth=1
    ),
    margin=dict(l=10, r=10, t=10, b=10),
    hovermode='x unified'
)

st.plotly_chart(curve_fig, use_container_width=True)

st.markdown("---")

# Original layout
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
    with st.expander("ℹ️ Gradient Logic", expanded=False):
        st.markdown("""
        **TOP Chart (Gamma Density) - Smooth Heat Map:**
        - 🟢 **BRIGHT GREEN** = Highest GEX concentration (strong resistance)
        - 🟡 **YELLOW-GREEN** = Medium-high gamma
        - 🟠 **ORANGE** = Medium-low gamma  
        - 🔴 **RED** = Lowest gamma (weak zones)
        
        **BOTTOM Chart (Call/Put Dominance):**
        - 🟡 **YELLOW** = Call gamma dominant (dealers short calls)
        - 🔵 **BLUE** = Put gamma dominant (dealers short puts)
        - Gray = Balanced
        
        *Uses cubic interpolation for smooth transitions like VolSignals*
        """)
    
    if prices is not None and len(prices) > 0:
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
        
        strikes = gex_filt['strike'].values
        call_gex_vals = gex_filt['call_gex'].values
        put_gex_vals = gex_filt['put_gex'].values
        total_gex = gex_filt['total_gex'].values
        
        # Set time range
        today = pd.Timestamp.now(tz='America/New_York').normalize()
        market_open = today + timedelta(hours=9, minutes=30)
        market_close = today + timedelta(hours=16)
        
        x_min = min(prices.index.min(), market_open) if not prices.empty else market_open
        x_max = max(prices.index.max(), market_close) if not prices.empty else market_close
        
        # CREATE SMOOTH GRADIENTS
        y_min = strikes.min()
        y_max = strikes.max()
        num_gradient_steps = 300
        y_grid = np.linspace(y_min, y_max, num_gradient_steps)
        
        # Interpolate GEX values for smooth transitions
        if len(strikes) > 2:
            f_total = interp1d(strikes, total_gex, kind='cubic', fill_value='extrapolate', bounds_error=False)
            total_gex_smooth = np.maximum(0, f_total(y_grid))
            
            f_call = interp1d(strikes, call_gex_vals, kind='cubic', fill_value='extrapolate', bounds_error=False)
            f_put = interp1d(strikes, put_gex_vals, kind='cubic', fill_value='extrapolate', bounds_error=False)
            call_gex_smooth = np.maximum(0, f_call(y_grid))
            put_gex_smooth = np.maximum(0, f_put(y_grid))
        else:
            total_gex_smooth = np.zeros(num_gradient_steps)
            call_gex_smooth = np.zeros(num_gradient_steps)
            put_gex_smooth = np.zeros(num_gradient_steps)
        
        max_gex_smooth = total_gex_smooth.max() if total_gex_smooth.max() > 0 else 1
        
        # TOP CHART: SMOOTH Gamma gradient
        for i in range(len(y_grid) - 1):
            gex_value = total_gex_smooth[i]
            normalized = gex_value / max_gex_smooth
            
            if normalized > 0.6:
                opacity = min(0.7, normalized * 0.8)
                color = f'rgba(0, 255, 0, {opacity})'
            elif normalized > 0.3:
                opacity = 0.5
                green_component = int(255 * (normalized - 0.3) / 0.3)
                color = f'rgba({255-green_component}, 255, 0, {opacity})'
            elif normalized > 0.1:
                opacity = 0.4
                color = f'rgba(255, 150, 0, {opacity})'
            else:
                opacity = 0.3
                color = f'rgba(255, 0, 0, {opacity})'
            
            fig2.add_shape(
                type="rect",
                x0=x_min,
                x1=x_max,
                y0=y_grid[i],
                y1=y_grid[i + 1],
                fillcolor=color,
                layer='below',
                line_width=0,
                row=1, col=1
            )
        
        # BOTTOM CHART: SMOOTH Call/Put dominance gradient
        for i in range(len(y_grid) - 1):
            call_val = call_gex_smooth[i]
            put_val = put_gex_smooth[i]
            total = call_val + put_val
            
            if total > 0:
                call_ratio = call_val / total
                put_ratio = put_val / total
                
                if call_ratio > 0.55:
                    opacity = min(0.7, call_ratio * 0.8)
                    color = f'rgba(255, 215, 0, {opacity})'
                elif put_ratio > 0.55:
                    opacity = min(0.7, put_ratio * 0.8)
                    color = f'rgba(74, 158, 255, {opacity})'
                else:
                    opacity = 0.3
                    color = f'rgba(165, 186, 128, {opacity})'
            else:
                color = 'rgba(50, 50, 50, 0.1)'
            
            fig2.add_shape(
                type="rect",
                x0=x_min,
                x1=x_max,
                y0=y_grid[i],
                y1=y_grid[i + 1],
                fillcolor=color,
                layer='below',
                line_width=0,
                row=2, col=1
            )
        
        # Add candlesticks
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
        
        fig2.update_xaxes(
            gridcolor='#1a1f3a',
            showgrid=True,
            range=[x_min, x_max],
            fixedrange=False
        )
        
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
st.markdown("---")
if prices is not None and len(prices) > 0:
    time_range = f"{prices.index.min().strftime('%H:%M')} - {prices.index.max().strftime('%H:%M')}"
    st.caption(f"Data: Barchart + YFinance | Price Range: {time_range} | Updated: {datetime.now().strftime('%H:%M:%S')}")
else:
    st.caption(f"Data: Barchart + YFinance | Updated: {datetime.now().strftime('%H:%M:%S')}")
