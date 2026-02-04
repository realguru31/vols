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
    # SPY can be on AMEX, NYSE, or NASDAQ depending on TradingView's data
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
    # Try multiple exchanges
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
            spot = data['meta']['quote'].get('lastPrice') or data['meta']['quote'].get('close')
        
        # If no spot in metadata, calculate from ATM options
        if spot is None:
            # Get all options data
            all_options = []
            for option_type, options in data['data'].items():
                all_options.extend(options)
            
            if all_options:
                # Find ATM strike (highest OI or volume)
                sorted_by_oi = sorted(all_options, key=lambda x: float(x.get('openInterest', 0)), reverse=True)
                if sorted_by_oi:
                    spot = float(sorted_by_oi[0].get('lastPrice', 0)) * 10  # Rough estimate
            
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
st.info("✅ **Data Sources:** Barchart (Options + Spot) | TradingView (Candlesticks) | Calculated (Expirations)")

# Controls
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
with col1:
    TICKER = st.text_input("Ticker", "SPY", key="ticker", label_visibility="collapsed").upper()
with col2:
    EXPIRY = st.number_input("Expiry", 0, 20, 1, key="expiry", label_visibility="collapsed", 
                             help="0=Current month, 1=Next month, etc. Calculated as 3rd Friday")
with col3:
    RANGE_PCT = st.slider("Range", 5, 30, 15, key="range", label_visibility="collapsed")
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

# Get candlestick price data (TradingView only - no rate limits!)
@st.cache_data(ttl=300)
def get_prices(ticker):
    try:
        tv = get_tv_datafeed()
        hist = get_tv_prices(tv, symbol=ticker, n_bars=100)
        
        if hist is None or hist.empty:
            return None
        
        # Filter to today if possible
        today = pd.Timestamp.now().normalize()
        hist_today = hist[hist.index.date >= today.date()]
        
        return hist_today if not hist_today.empty else hist
    except Exception as e:
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
    st.markdown("#### GEX Zones + Price Action")
    st.caption(f"Drawing {len(gex_filt)-1} colored zones based on GEX intensity")
    
    if prices is not None and len(prices) > 0:
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                             specs=[[{"secondary_y": False}], [{"secondary_y": False}]])
        
        strikes = gex_filt['strike'].values
        call_gex_vals = gex_filt['call_gex'].values
        put_gex_vals = gex_filt['put_gex'].values
        total_gex = gex_filt['total_gex'].values
        
        # Full trading day range
        today = pd.Timestamp.now(tz='America/New_York').normalize()
        market_open = today + timedelta(hours=9, minutes=30)
        market_close = today + timedelta(hours=16)
        x_min = min(prices.index.min(), market_open) if not prices.empty else market_open
        x_max = max(prices.index.max(), market_close) if not prices.empty else market_close
        
        # Smooth GEX using Gaussian filter
        if len(strikes) > 3:
            total_gex_smooth = gaussian_filter1d(total_gex, sigma=1.5)
            call_gex_smooth = gaussian_filter1d(call_gex_vals, sigma=1.5)
            put_gex_smooth = gaussian_filter1d(put_gex_vals, sigma=1.5)
        else:
            total_gex_smooth = total_gex
            call_gex_smooth = call_gex_vals
            put_gex_smooth = put_gex_vals
        
        max_gex = total_gex_smooth.max() if total_gex_smooth.max() > 0 else 1
        
        # TEST: Add one very obvious zone to verify hrect works
        mid_strike = strikes[len(strikes)//2]
        fig2.add_hrect(
            y0=mid_strike - 2,
            y1=mid_strike + 2,
            fillcolor='rgba(255, 0, 255, 0.5)',  # Bright magenta test zone
            layer='below',
            line_width=0,
            annotation_text="TEST ZONE",
            row=1, col=1
        )
        
        # TOP CHART: GEX-colored background zones + candlesticks
        # Create filled areas based on GEX intensity
        for i in range(len(strikes) - 1):
            intensity = total_gex_smooth[i] / max_gex
            
            # Color by GEX intensity - MUCH MORE VISIBLE
            if intensity > 0.6:
                color = 'rgba(0, 255, 0, 0.4)'  # Green - high GEX (was 0.15)
            elif intensity > 0.3:
                color = 'rgba(255, 215, 0, 0.35)'  # Yellow - medium (was 0.12)
            else:
                color = 'rgba(255, 0, 0, 0.3)'  # Red - low GEX (was 0.1)
            
            # Add colored horizontal zone
            fig2.add_hrect(
                y0=strikes[i],
                y1=strikes[i+1],
                fillcolor=color,
                layer='below',
                line_width=0,
                row=1, col=1
            )
        
        # Add candlesticks on top
        fig2.add_trace(
            go.Candlestick(
                x=prices.index,
                open=prices['Open'],
                high=prices['High'],
                low=prices['Low'],
                close=prices['Close'],
                increasing_line_color='#00FF00',
                decreasing_line_color='#FF0000',
                increasing_fillcolor='rgba(0,255,0,0.2)',  # More transparent candles
                decreasing_fillcolor='rgba(255,0,0,0.2)',
                line=dict(width=1),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # BOTTOM CHART: Call/Put dominance zones + candlesticks
        for i in range(len(strikes) - 1):
            if call_gex_smooth[i] > put_gex_smooth[i]:
                color = 'rgba(255, 215, 0, 0.4)'  # Yellow - call heavy (was 0.15)
            else:
                color = 'rgba(74, 158, 255, 0.4)'  # Blue - put heavy (was 0.15)
            
            fig2.add_hrect(
                y0=strikes[i],
                y1=strikes[i+1],
                fillcolor=color,
                layer='below',
                line_width=0,
                row=2, col=1
            )
        
        # Add candlesticks
        fig2.add_trace(
            go.Candlestick(
                x=prices.index,
                open=prices['Open'],
                high=prices['High'],
                low=prices['Low'],
                close=prices['Close'],
                increasing_line_color='#00FF00',
                decreasing_line_color='#FF0000',
                increasing_fillcolor='rgba(0,255,0,0.2)',
                decreasing_fillcolor='rgba(255,0,0,0.2)',
                line=dict(width=1),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Add spot lines to both charts
        for row in [1, 2]:
            fig2.add_hline(
                y=spot,
                line=dict(color='#FFD700', width=2, dash='dot'),
                row=row, col=1
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
if prices is not None and len(prices) > 0:
    time_range = f"{prices.index.min().strftime('%H:%M')} - {prices.index.max().strftime('%H:%M')}"
    st.caption(f"Barchart (Options+Spot) | TradingView (Candles) | Range: {time_range} | {datetime.now().strftime('%H:%M:%S')}")
else:
    st.caption(f"Barchart (Options+Spot) | TradingView (Candles) | {datetime.now().strftime('%H:%M:%S')}")
