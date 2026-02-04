import streamlit as st
import requests
import pandas as pd
from urllib.parse import unquote
from datetime import datetime, timedelta
from tvDatafeed import TvDatafeed, Interval
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
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

# Initialize TradingView datafeed (for candlestick data only)
@st.cache_resource
def get_tv_datafeed():
    """Initialize TradingView datafeed (cached)"""
    return TvDatafeed()

# Helper to get spot price from TradingView (fallback only)
def get_tv_spot_price(tv, symbol="SPY"):
    """Get spot from TradingView as fallback"""
    exchanges_to_try = ['AMEX', 'NYSE', 'NASDAQ']
    
    for exchange in exchanges_to_try:
        try:
            df = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_1_minute, n_bars=1)
            if df is not None and not df.empty:
                return df['close'].iloc[-1]
        except:
            continue
    return None

# Helper to get candlesticks from TradingView
def get_tv_prices(tv, symbol="SPY", n_bars=100):
    """Get 5-min candlestick data from TradingView"""
    exchanges_to_try = ['AMEX', 'NYSE', 'NASDAQ']
    
    for exchange in exchanges_to_try:
        try:
            df = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_5_minute, n_bars=n_bars)
            if df is not None and not df.empty:
                df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
                return df
        except:
            continue
    return None

# Calculate options expiration dates (3rd Friday of month)
def get_options_expirations(num_expirations=20):
    """Calculate standard monthly options expirations"""
    expirations = []
    current_date = datetime.now()
    
    for month_offset in range(num_expirations):
        year = current_date.year + (current_date.month + month_offset - 1) // 12
        month = (current_date.month + month_offset - 1) % 12 + 1
        
        # Find 3rd Friday
        first_day = datetime(year, month, 1)
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + timedelta(weeks=2)
        
        # Only include future dates
        if third_friday >= current_date:
            expirations.append(third_friday.strftime("%Y-%m-%d"))
    
    return expirations

# Fetch Barchart data
@st.cache_data(ttl=600)
def fetch_barchart_options(ticker_symbol, expiry_offset=0):
    try:
        # Calculate expiration dates (3rd Friday of each month)
        expiry_dates = get_options_expirations()
        
        if not expiry_dates:
            return {'error': 'Failed to calculate expiration dates', 'type': 'NoOptionsError', 'traceback': 'Expiration calculation failed'}
        
        if expiry_offset >= len(expiry_dates):
            return {'error': f'Expiry offset {expiry_offset} too high, only {len(expiry_dates)} dates available', 'type': 'OffsetError', 'traceback': f'Available dates: {expiry_dates[:5]}'}
        
        expiry = expiry_dates[expiry_offset]
        
        # Barchart API setup
        geturl = f'https://www.barchart.com/etfs-funds/quotes/{ticker_symbol}/volatility-greeks'
        apiurl = 'https://www.barchart.com/proxies/core-api/v1/options/get'
        
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
        
        # Try to get spot price from Barchart metadata
        spot = None
        if 'meta' in data and 'quote' in data['meta']:
            spot_raw = data['meta']['quote'].get('lastPrice') or data['meta']['quote'].get('close')
            if spot_raw:
                # Remove commas and convert to float
                spot = float(str(spot_raw).replace(',', ''))
        
        # If no spot in metadata, calculate from ATM options
        if spot is None:
            # Get all options data
            all_options = []
            for option_type, options in data['data'].items():
                all_options.extend(options)
            
            if all_options:
                # Find ATM strike (highest OI or volume)
                sorted_by_oi = sorted(all_options, key=lambda x: float(str(x.get('openInterest', 0)).replace(',', '')), reverse=True)
                if sorted_by_oi:
                    # Use strike price of highest OI option as proxy for spot
                    strike_str = sorted_by_oi[0].get('strikePrice', 0)
                    spot = float(str(strike_str).replace(',', ''))
            
            # Fallback: use TradingView for spot only
            if spot is None or spot == 0:
                tv = get_tv_datafeed()
                spot = get_tv_spot_price(tv, ticker_symbol)
                
            if spot is None:
                return {'error': 'Could not determine spot price', 'type': 'SpotPriceError', 'traceback': 'No spot in Barchart data and TradingView failed'}
        
        data_list = []
        for option_type, options in data['data'].items():
            for option in options:
                option['optionType'] = option_type
                data_list.append(option)
        
        if not data_list:
            return {'error': 'No options data returned from API', 'type': 'EmptyDataError', 'traceback': 'API returned empty options list'}
        
        df = pd.DataFrame(data_list)
        
        # Strip commas from numeric fields before conversion
        numeric_fields = ['strikePrice', 'openInterest', 'gamma', 'theta', 'lastPrice', 'volume']
        for field in numeric_fields:
            if field in df.columns:
                df[field] = df[field].astype(str).str.replace(',', '')
        
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

# Info banner
st.info("✅ **Data:** Barchart (Options + Spot) | TradingView (Candles - optional) | Calculated (Expirations)")

# Controls
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
with col1:
    TICKER = st.text_input("Ticker", "SPY", key="ticker", label_visibility="collapsed").upper()
with col2:
    EXPIRY = st.number_input("Expiry", 0, 20, 1, key="expiry", label_visibility="collapsed", 
                             help="0=Current month, 1=Next month, etc. Calculated as 3rd Friday")
with col3:
    RANGE_PCT = st.slider("Range", 2, 5, 3, key="range", label_visibility="collapsed")
with col4:
    if st.button("🔄"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# Fetch data
result = fetch_barchart_options(TICKER, EXPIRY)

# Check for errors
if not result:
    st.error("❌ Failed to fetch data - returned None")
    st.stop()

if 'error' in result:
    st.error("❌ Failed to fetch data")
    st.error(f"**Error Type:** {result['type']}")
    st.error(f"**Error Message:** {result['error']}")
    
    with st.expander("🔍 Technical Details"):
        st.code(result['traceback'])
    st.stop()

spot = result['spot']
expiry = result['expiry']
gex_df = compute_gex(result['calls'], result['puts'])

# Debug info
with st.expander("🔍 Debug Info", expanded=False):
    st.write(f"**Spot Price:** ${spot:.2f}")
    st.write(f"**Expiration:** {expiry}")
    st.write(f"**Total Options:** {len(result['calls'])} calls, {len(result['puts'])} puts")

# Filter for price range only - keep ALL strikes including zero GEX
price_range = spot * (RANGE_PCT / 100)
gex_filt = gex_df[
    (gex_df['strike'] >= spot - price_range) &
    (gex_df['strike'] <= spot + price_range)
]
# Don't filter out zero GEX - we want all $1 strikes!

if len(gex_filt) == 0:
    st.warning("No data")
    st.stop()

# Get candlestick price data (TradingView)
@st.cache_data(ttl=300)
def get_prices(ticker):
    """Fetch price data from TradingView"""
    try:
        tv = get_tv_datafeed()
        hist = get_tv_prices(tv, symbol=ticker, n_bars=100)
        
        if hist is None or hist.empty:
            return None
        
        # Filter to today if possible
        today = pd.Timestamp.now().normalize()
        hist_today = hist[hist.index.date >= today.date()]
        
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
    
    st.plotly_chart(fig1, width='stretch', key='gex_bars_chart')

with col_right:
    st.markdown("#### GEX Profile + Price Action")
    
    if prices is not None and len(prices) > 0:
        st.caption(f"GEX curves overlaid on price chart (strikes: {len(gex_filt)})")
        
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.05)
        
        strikes = gex_filt['strike'].values
        call_gex_vals = gex_filt['call_gex'].values
        put_gex_vals = gex_filt['put_gex'].values
        total_gex = gex_filt['total_gex'].values
        
        # Smooth GEX curves using Gaussian filter (like matplotlib version)
        if len(strikes) > 3:
            total_gex_smooth = gaussian_filter1d(total_gex, sigma=1.5)
            call_gex_smooth = gaussian_filter1d(call_gex_vals, sigma=1.5)
            put_gex_smooth = gaussian_filter1d(put_gex_vals, sigma=1.5)
        else:
            total_gex_smooth = total_gex
            call_gex_smooth = call_gex_vals
            put_gex_smooth = put_gex_vals
        
        # Get max GEX for scaling
        max_gex = max(total_gex_smooth.max(), call_gex_smooth.max(), put_gex_smooth.max())
        if max_gex == 0:
            max_gex = 1
        
        # Calculate time range for GEX scaling
        time_min = prices.index.min()
        time_max = prices.index.max()
        time_range = time_max - time_min
        
        # TOP CHART: Total GEX Profile + Candlesticks
        # Add candlesticks first
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
                name='Price',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add GEX curves as VERTICAL profiles (rotated 90 degrees)
        # Curves extend RIGHT to LEFT, starting immediately after last candle
        gex_x_offset = time_max + time_range * 0.02  # Start just 2% beyond candles
        gex_x_scale = (time_range * 0.25) / max_gex  # Scale to fit in 25% of time range
        
        # Total GEX (orange curve) - NO FILL
        gex_x_total = [gex_x_offset - (val * gex_x_scale) for val in total_gex_smooth]
        fig2.add_trace(
            go.Scatter(
                x=gex_x_total,
                y=strikes,
                mode='lines',
                line=dict(color='orange', width=3),
                name='Total GEX',
                hovertemplate='Strike: %{y:.0f}<br>GEX: %{customdata:,.0f}<extra></extra>',
                customdata=total_gex_smooth,
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Call GEX (green curve) - NO FILL
        gex_x_calls = [gex_x_offset - (val * gex_x_scale) for val in call_gex_smooth]
        fig2.add_trace(
            go.Scatter(
                x=gex_x_calls,
                y=strikes,
                mode='lines',
                line=dict(color='green', width=2.5),
                name='Call GEX',
                hovertemplate='Strike: %{y:.0f}<br>Call GEX: %{customdata:,.0f}<extra></extra>',
                customdata=call_gex_smooth,
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Put GEX (red curve) - NO FILL
        gex_x_puts = [gex_x_offset - (val * gex_x_scale) for val in put_gex_smooth]
        fig2.add_trace(
            go.Scatter(
                x=gex_x_puts,
                y=strikes,
                mode='lines',
                line=dict(color='red', width=2.5),
                name='Put GEX',
                hovertemplate='Strike: %{y:.0f}<br>Put GEX: %{customdata:,.0f}<extra></extra>',
                customdata=put_gex_smooth,
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Spot price line with better annotation
        fig2.add_hline(
            y=spot,
            line=dict(color='#FFD700', width=2, dash='dot'),
            annotation=dict(
                text=f"${spot:.2f}",
                xref="paper",
                x=0.02,  # Left side, 2% from edge
                font=dict(color='#FFD700', size=12)
            ),
            row=1, col=1
        )
        
        # BOTTOM CHART: Net GEX Profile + Candlesticks
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
                name='Price',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Net GEX (call - put) - GOLD curve - NO FILL
        net_gex_smooth = call_gex_smooth - put_gex_smooth
        max_net = max(abs(net_gex_smooth.min()), abs(net_gex_smooth.max()))
        if max_net == 0:
            max_net = 1
        
        net_gex_x_scale = (time_range * 0.25) / max_net
        gex_x_net = [gex_x_offset - (val * net_gex_x_scale) for val in net_gex_smooth]
        
        fig2.add_trace(
            go.Scatter(
                x=gex_x_net,
                y=strikes,
                mode='lines',
                line=dict(color='gold', width=3),
                name='Net GEX',
                hovertemplate='Strike: %{y:.0f}<br>Net GEX: %{customdata:,.0f}<extra></extra>',
                customdata=net_gex_smooth,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Spot line with better annotation
        fig2.add_hline(
            y=spot,
            line=dict(color='#FFD700', width=2, dash='dot'),
            annotation=dict(
                text=f"${spot:.2f}",
                xref="paper",
                x=0.02,  # Left side
                font=dict(color='#FFD700', size=12)
            ),
            row=2, col=1
        )
        
        fig2.update_layout(
            height=850,
            plot_bgcolor='#0a0e27',
            paper_bgcolor='#0a0e27',
            font=dict(color='#fff', size=10),
            showlegend=False,
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=10, b=10),
            hovermode='closest'
        )
        
        # Set X-axis to show curves right after candles
        x_range_extended = time_max + time_range * 0.05  # Just 5% padding on right
        fig2.update_xaxes(
            gridcolor='#1a1f3a',
            showgrid=True,
            range=[time_min, x_range_extended],
            fixedrange=False
        )
        
        # Set Y-axis ranges
        y_padding = (strikes.max() - strikes.min()) * 0.1
        fig2.update_yaxes(
            gridcolor='#1a1f3a',
            tickformat='$.0f',
            range=[strikes.min() - y_padding, strikes.max() + y_padding],
            row=1, col=1
        )
        fig2.update_yaxes(
            gridcolor='#1a1f3a',
            tickformat='$.0f',
            range=[strikes.min() - y_padding, strikes.max() + y_padding],
            row=2, col=1
        )
        
        st.plotly_chart(fig2, width='stretch', key='gex_overlay_chart')
    else:
        st.warning("⚠️ Price data unavailable (TradingView connection issue)")
        st.info("GEX analysis is still working - only price overlay is affected. Try refreshing or check back later.")

# Footer
if prices is not None and len(prices) > 0:
    time_range = f"{prices.index.min().strftime('%H:%M')} - {prices.index.max().strftime('%H:%M')}"
    st.caption(f"Barchart (Options+Spot) | TradingView (Candles) | Range: {time_range} | {datetime.now().strftime('%H:%M:%S')}")
else:
    st.caption(f"Barchart (Options+Spot) | TradingView (Candles) | {datetime.now().strftime('%H:%M:%S')}")
