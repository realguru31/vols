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
st.markdown("## 📊 GEX Profile Analyzer - VolSignals Style")

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
        # Create gradient-filled area chart (like your examples)
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
        
        strikes = gex_filt['strike'].values
        total_gex = gex_filt['total_gex'].values
        net_gex = gex_filt['net_gex'].values
        
        # Create smooth interpolated curves
        if len(strikes) > 3:
            # Create fine grid for smooth interpolation
            fine_strikes = np.linspace(strikes.min(), strikes.max(), 300)
            
            # Interpolate total GEX
            f_total = interp1d(strikes, total_gex, kind='cubic', fill_value=0, bounds_error=False)
            total_gex_smooth = np.maximum(0, f_total(fine_strikes))
            
            # Interpolate net GEX
            f_net = interp1d(strikes, net_gex, kind='cubic', fill_value=0, bounds_error=False)
            net_gex_smooth = f_net(fine_strikes)
        else:
            fine_strikes = strikes
            total_gex_smooth = total_gex
            net_gex_smooth = net_gex
        
        # TOP CHART: Gamma Density Gradient (like first example image)
        # Create gradient effect using multiple filled areas with varying opacity
        
        # Normalize for gradient effect
        max_gamma = total_gex_smooth.max() if total_gex_smooth.max() > 0 else 1
        
        # Create gradient by stacking multiple semi-transparent areas
        num_gradient_layers = 10
        for i in range(num_gradient_layers):
            # Calculate opacity based on layer (darker at high gamma)
            opacity = 0.3 * (i + 1) / num_gradient_layers
            
            # Create threshold for this layer
            threshold = max_gamma * (i / num_gradient_layers)
            
            # Create filled area for this gradient layer
            layer_values = np.where(total_gex_smooth >= threshold, total_gex_smooth, 0)
            
            # Create gradient color from red to green based on gamma intensity
            intensity = i / num_gradient_layers
            if intensity > 0.6:
                # High gamma = GREEN
                color = f'rgba(0, 255, 0, {opacity})'
            elif intensity > 0.3:
                # Medium = YELLOW/GREEN blend
                green_component = int(255 * (intensity - 0.3) / 0.3)
                color = f'rgba({255-green_component}, 255, 0, {opacity})'
            elif intensity > 0.1:
                # Low-medium = ORANGE
                color = f'rgba(255, 150, 0, {opacity})'
            else:
                # Very low = RED
                color = f'rgba(255, 0, 0, {opacity})'
            
            # Add filled area for this gradient layer
            fig2.add_trace(
                go.Scatter(
                    x=prices.index,
                    y=[fine_strikes[np.argmax(layer_values)]] * len(prices.index),
                    fill='toself',
                    fillcolor=color,
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
        
        # BOTTOM CHART: Call/Put Dominance Gradient
        # Create gradient for net GEX (blue for puts, yellow/gold for calls)
        
        # Find min and max for normalization
        max_abs_net = max(abs(net_gex_smooth.min()), abs(net_gex_smooth.max())) if len(net_gex_smooth) > 0 else 1
        
        num_dominance_layers = 10
        for i in range(num_dominance_layers):
            opacity = 0.3 * (i + 1) / num_dominance_layers
            
            # Create threshold for this layer
            threshold_abs = max_abs_net * (i / num_dominance_layers)
            
            # Create mask for this layer
            if max_abs_net > 0:
                layer_mask = abs(net_gex_smooth) >= threshold_abs
                layer_net = np.where(layer_mask, net_gex_smooth, 0)
            else:
                layer_net = np.zeros_like(net_gex_smooth)
            
            # Determine color based on sign (calls vs puts)
            avg_sign = np.sign(np.mean(layer_net[layer_mask])) if np.any(layer_mask) else 0
            
            if avg_sign > 0.1:
                # Call dominant = YELLOW/GOLD
                color = f'rgba(255, 215, 0, {opacity})'
            elif avg_sign < -0.1:
                # Put dominant = BLUE
                color = f'rgba(74, 158, 255, {opacity})'
            else:
                # Balanced = blend
                color = f'rgba(165, 186, 128, {opacity})'
            
            # Add filled area
            if np.any(layer_mask):
                dominant_strike = fine_strikes[np.argmax(abs(layer_net))]
                fig2.add_trace(
                    go.Scatter(
                        x=prices.index,
                        y=[dominant_strike] * len(prices.index),
                        fill='toself',
                        fillcolor=color,
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=2, col=1
                )
        
        # Add candlesticks on top
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
        
        # Add gamma curve on top of gradient
        fig2.add_trace(
            go.Scatter(
                x=prices.index,
                y=[fine_strikes[np.argmax(total_gex_smooth)]] * len(prices.index),
                mode='lines',
                line=dict(color='white', width=1, dash='dash'),
                name='Gamma Peak',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Add net GEX curve
        fig2.add_trace(
            go.Scatter(
                x=prices.index,
                y=[fine_strikes[np.argmax(abs(net_gex_smooth))]] * len(prices.index),
                mode='lines',
                line=dict(color='cyan', width=1, dash='dash'),
                name='Net GEX Peak',
                showlegend=True
            ),
            row=2, col=1
        )
        
        fig2.update_layout(
            height=850,
            plot_bgcolor='#0a0e27',
            paper_bgcolor='#0a0e27',
            font=dict(color='#fff', size=10),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(10, 14, 39, 0.8)'
            ),
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=10, b=10),
            hovermode='x unified'
        )
        
        # Set y-axis ranges
        fig2.update_yaxes(
            gridcolor='#1a1f3a',
            tickformat='$.0f',
            range=[strikes.min() - 5, strikes.max() + 5],
            row=1, col=1,
            title="Gamma"
        )
        fig2.update_yaxes(
            gridcolor='#1a1f3a',
            tickformat='$.0f',
            range=[strikes.min() - 5, strikes.max() + 5],
            row=2, col=1,
            title="Charm/Net GEX"
        )
        
        # Set x-axis time format
        fig2.update_xaxes(
            gridcolor='#1a1f3a',
            tickformat="%H:%M",
            row=2, col=1
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Add explanation
        with st.expander("ℹ️ Gradient Logic", expanded=False):
            st.markdown("""
            **TOP Chart (Gamma Density) - Gradient Fill:**
            - 🟢 **DARK GREEN** = Highest GEX concentration (strong resistance)
            - 🟡 **YELLOW-GREEN** = Medium-high gamma
            - 🟠 **ORANGE** = Medium-low gamma  
            - 🔴 **LIGHT RED** = Lowest gamma (weak zones)
            
            **BOTTOM Chart (Call/Put Dominance):**
            - 🟡 **GOLD/YELLOW** = Call gamma dominant (dealers short calls = resistance)
            - 🔵 **BLUE** = Put gamma dominant (dealers short puts = support)
            - 🟢 **GREEN** = Balanced gamma
            
            **White dashed line** = Gamma peak (highest concentration)
            **Cyan dashed line** = Net GEX peak
            
            *Creates smooth gradient effects by stacking multiple semi-transparent filled areas*
            """)
    else:
        st.warning("No price data available for gradient visualization")

# Footer
if prices is not None and len(prices) > 0:
    time_range = f"{prices.index.min().strftime('%H:%M')} - {prices.index.max().strftime('%H:%M')}"
    st.caption(f"Data: Barchart + YFinance | Price Range: {time_range} | Updated: {datetime.now().strftime('%H:%M:%S')}")
else:
    st.caption(f"Data: Barchart + YFinance | Updated: {datetime.now().strftime('%H:%M:%S')}")
