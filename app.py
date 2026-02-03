import streamlit as st
import requests
import pandas as pd
from urllib.parse import unquote
from datetime import datetime
import yfinance as yf
import plotly.graph_objects as go
from math import log, sqrt
from scipy.stats import norm
import numpy as np

# Page config
st.set_page_config(page_title="GEX Profile Analyzer", layout="wide", initial_sidebar_state="expanded")

# Black-Scholes (backup for IV if needed)
def bs_greeks(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0:
        return 0, 0
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    delta = norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * sqrt(T))
    return delta, gamma

# Fetch options from Barchart
@st.cache_data(ttl=300)
def fetch_barchart_options(ticker_symbol, expiry_offset=0):
    """Fetch options data from Barchart"""
    try:
        # Get expiration dates from yfinance
        ticker_yf = yf.Ticker(ticker_symbol)
        expiry_dates = ticker_yf.options
        
        if not expiry_dates or expiry_offset >= len(expiry_dates):
            return None
        
        expiry = expiry_dates[expiry_offset]
        
        # Get spot price
        spot = ticker_yf.history(period="1d")['Close'].iloc[-1]
        
        # Barchart URLs and headers
        geturl = f'https://www.barchart.com/etfs-funds/quotes/{ticker_symbol}/volatility-greeks'
        apiurl = 'https://www.barchart.com/proxies/core-api/v1/options/get'
        
        getheaders = {
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'accept-encoding': 'gzip, deflate, br',
            'accept-language': 'en-US,en;q=0.9',
            'cache-control': 'max-age=0',
            'upgrade-insecure-requests': '1',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Initial session request
        s = requests.Session()
        r = s.get(geturl, headers=getheaders, timeout=10)
        r.raise_for_status()
        
        # API headers with XSRF token
        headers = {
            'accept': 'application/json',
            'accept-encoding': 'gzip, deflate, br',
            'accept-language': 'en-US,en;q=0.9',
            'referer': geturl,
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'x-xsrf-token': unquote(unquote(s.cookies.get_dict()['XSRF-TOKEN']))
        }
        
        # API payload
        payload = {
            'baseSymbol': ticker_symbol,
            'groupBy': 'optionType',
            'expirationDate': expiry,
            'orderBy': 'strikePrice',
            'orderDir': 'asc',
            'raw': '1',
            'fields': 'symbol,strikePrice,lastPrice,volatility,delta,gamma,volume,openInterest,optionType'
        }
        
        # Get options data
        r = s.get(apiurl, params=payload, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        # Extract and combine data
        data_list = []
        for option_type, options in data['data'].items():
            for option in options:
                option['optionType'] = option_type
                data_list.append(option)
        
        # Create DataFrame
        df = pd.DataFrame(data_list)
        
        # Clean and convert data
        df['strikePrice'] = pd.to_numeric(df['strikePrice'], errors='coerce')
        df['openInterest'] = pd.to_numeric(df['openInterest'], errors='coerce').fillna(0).astype(int)
        df['gamma'] = pd.to_numeric(df['gamma'], errors='coerce').fillna(0)
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)
        
        # Clean volatility (remove % and convert)
        if 'volatility' in df.columns:
            df['volatility'] = df['volatility'].astype(str).str.replace('%', '').str.strip()
            df['volatility'] = pd.to_numeric(df['volatility'], errors='coerce') / 100
        
        # Separate calls and puts
        calls = df[df['optionType'] == 'Call'].copy()
        puts = df[df['optionType'] == 'Put'].copy()
        
        return {
            'spot': spot,
            'expiry': expiry,
            'expiry_dates': expiry_dates,
            'calls': calls,
            'puts': puts
        }
        
    except Exception as e:
        st.error(f"Error fetching Barchart data: {str(e)}")
        return None

# App title
st.title("🎯 GEX Profile Analyzer")
st.markdown("*Professional dealer gamma exposure analysis powered by Barchart*")

# Sidebar
st.sidebar.header("⚙️ Settings")
TICKER = st.sidebar.text_input("Ticker Symbol", "SPY").upper()
EXPIRY_OFFSET = st.sidebar.number_input("Expiry Index (0=nearest)", min_value=0, max_value=20, value=1, step=1)
price_range_pct = st.sidebar.slider("Strike Range (%)", min_value=1, max_value=20, value=10)

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("💡 **Barchart Data:** OI updates more frequently than Yahoo Finance, works better outside market hours!")

# Fetch data
with st.spinner("🔍 Fetching options data from Barchart..."):
    result = fetch_barchart_options(TICKER, EXPIRY_OFFSET)
    
    if result is None:
        st.error(f"❌ Could not fetch options data for {TICKER}")
        st.info("Try: 1) Different expiry index, 2) Check ticker symbol, 3) Refresh page")
        st.stop()
    
    spot = result['spot']
    expiry = result['expiry']
    expiry_dates = result['expiry_dates']
    calls = result['calls']
    puts = result['puts']

# Calculate DTE
expiry_dt = datetime.strptime(expiry, "%Y-%m-%d")
days_to_expiry = (expiry_dt - datetime.now()).days

# Header metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Spot", f"${spot:.2f}")
with col2:
    st.metric("Ticker", TICKER)
with col3:
    st.metric("DTE", f"{days_to_expiry}d")
with col4:
    st.metric("Expiry", expiry.split('-')[1] + '/' + expiry.split('-')[2])

# Data quality check
calls_with_oi = calls[calls['openInterest'] > 0]
puts_with_oi = puts[puts['openInterest'] > 0]
oi_coverage = (len(calls_with_oi) + len(puts_with_oi)) / (len(calls) + len(puts)) * 100 if len(calls) + len(puts) > 0 else 0

st.info(f"📊 Data Quality: {oi_coverage:.1f}% of options have OI > 0 | {len(calls)} calls, {len(puts)} puts | Source: Barchart")

if oi_coverage < 5:
    st.warning("⚠️ Low OI coverage. Try a different expiry index for better data.")

# Compute GEX using Barchart's gamma
def compute_gex_barchart():
    """Compute GEX using gamma from Barchart"""
    all_strikes = sorted(list(set(calls['strikePrice'].tolist() + puts['strikePrice'].tolist())))
    gex_by_strike = []
    
    for K in all_strikes:
        if pd.isna(K):
            continue
            
        call_gex = 0
        put_gex = 0
        call_oi = 0
        put_oi = 0
        call_gamma = 0
        put_gamma = 0
        
        # Calls
        call_data = calls[calls['strikePrice'] == K]
        if not call_data.empty:
            row = call_data.iloc[0]
            call_oi = int(row['openInterest'])
            call_gamma = float(row['gamma'])
            if call_oi > 0 and call_gamma > 0:
                call_gex = call_gamma * call_oi * 100  # Standard GEX formula
        
        # Puts
        put_data = puts[puts['strikePrice'] == K]
        if not put_data.empty:
            row = put_data.iloc[0]
            put_oi = int(row['openInterest'])
            put_gamma = float(row['gamma'])
            if put_oi > 0 and put_gamma > 0:
                put_gex = put_gamma * put_oi * 100
        
        net_gex = call_gex - put_gex
        gex_by_strike.append({
            'strike': K,
            'call_gex': call_gex,
            'put_gex': put_gex,
            'net_gex': net_gex,
            'call_oi': call_oi,
            'put_oi': put_oi,
            'call_gamma': call_gamma,
            'put_gamma': put_gamma,
            'total_gamma': abs(call_gex) + abs(put_gex)
        })
    
    return pd.DataFrame(gex_by_strike)

gex_df = compute_gex_barchart()

# Filter strikes
price_range = spot * (price_range_pct / 100)
min_strike = spot - price_range
max_strike = spot + price_range
gex_df_filtered = gex_df[(gex_df['strike'] >= min_strike) & (gex_df['strike'] <= max_strike)].copy()
gex_df_filtered = gex_df_filtered[(gex_df_filtered['call_gex'] != 0) | (gex_df_filtered['put_gex'] != 0)]

if len(gex_df_filtered) == 0:
    st.error("⚠️ No GEX data in selected range")
    st.info(f"Computed {len(gex_df)} strikes, none in ${min_strike:.0f}-${max_strike:.0f} with GEX. Try increasing strike range %.")
    st.stop()

# Summary metrics
total_call_gex = gex_df_filtered['call_gex'].sum()
total_put_gex = gex_df_filtered['put_gex'].sum()
net_gex = total_call_gex - total_put_gex

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Call GEX", f"{total_call_gex:,.0f}", help="Dealers SHORT gamma")
with col2:
    st.metric("Put GEX", f"{total_put_gex:,.0f}", help="Dealers LONG gamma")
with col3:
    gex_sign = "🟢" if net_gex > 0 else "🔴"
    st.metric("Net GEX", f"{gex_sign} {net_gex:,.0f}")

# Market regime
if net_gex > 0:
    st.success("**📊 POSITIVE GEX:** Dampening regime - dealers hedge by buying dips/selling rips → Lower volatility expected")
else:
    st.error("**⚡ NEGATIVE GEX:** Amplifying regime - dealers hedge by selling dips/buying rips → Higher volatility expected")

st.markdown("---")

# Main charts
col_chart1, col_chart2 = st.columns([1, 1])

with col_chart1:
    st.subheader("📊 Gamma Profile by Strike")
    
    fig1 = go.Figure()
    
    fig1.add_trace(go.Bar(
        y=gex_df_filtered['strike'],
        x=gex_df_filtered['call_gex'],
        orientation='h',
        name='Call GEX',
        marker=dict(color='rgba(0, 255, 0, 0.6)', line=dict(color='rgba(0, 200, 0, 1)', width=1)),
        hovertemplate='<b>$%{y:.0f}</b><br>Call GEX: %{x:,.0f}<br>OI: %{customdata[0]:,.0f}<extra></extra>',
        customdata=gex_df_filtered[['call_oi']].values
    ))
    
    fig1.add_trace(go.Bar(
        y=gex_df_filtered['strike'],
        x=-gex_df_filtered['put_gex'],
        orientation='h',
        name='Put GEX',
        marker=dict(color='rgba(255, 0, 0, 0.6)', line=dict(color='rgba(200, 0, 0, 1)', width=1)),
        hovertemplate='<b>$%{y:.0f}</b><br>Put GEX: %{x:,.0f}<br>OI: %{customdata[0]:,.0f}<extra></extra>',
        customdata=gex_df_filtered[['put_oi']].values
    ))
    
    fig1.add_hline(y=spot, line_dash="dash", line_color="cyan", line_width=3,
                   annotation=dict(text=f"${spot:.2f}", font=dict(size=12, color="cyan"), bgcolor="rgba(0,0,0,0.8)"),
                   annotation_position="right")
    
    if len(gex_df_filtered) > 0:
        max_gex_idx = gex_df_filtered['net_gex'].abs().idxmax()
        max_gex_strike = gex_df_filtered.loc[max_gex_idx, 'strike']
        fig1.add_hline(y=max_gex_strike, line_dash="dot", line_color="gold", line_width=2,
                       annotation=dict(text=f"${max_gex_strike:.0f}", font=dict(size=10, color="gold"), bgcolor="rgba(0,0,0,0.7)"),
                       annotation_position="left")
    
    fig1.update_layout(
        barmode='overlay',
        height=600,
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='white', size=11),
        xaxis=dict(title="Gamma Exposure", gridcolor='#1f1f1f', showgrid=True, tickformat=','),
        yaxis=dict(title="Strike", gridcolor='#1f1f1f', tickformat='$.0f'),
        hovermode='closest',
        showlegend=True,
        legend=dict(bgcolor='rgba(0,0,0,0.5)', bordercolor='white', borderwidth=1, font=dict(size=10)),
        margin=dict(l=10, r=10, t=10, b=10)
    )
    
    st.plotly_chart(fig1, use_container_width=True)

with col_chart2:
    st.subheader("📈 Price Action with Gamma Zones")
    
    # Get historical price
    try:
        ticker_yf = yf.Ticker(TICKER)
        hist_data = ticker_yf.history(period="5d", interval="5m")
        
        if not hist_data.empty:
            support_level = gex_df_filtered.loc[gex_df_filtered['put_gex'].idxmax(), 'strike']
            resistance_level = gex_df_filtered.loc[gex_df_filtered['call_gex'].idxmax(), 'strike']
            
            fig2 = go.Figure()
            
            fig2.add_trace(go.Candlestick(
                x=hist_data.index,
                open=hist_data['Open'],
                high=hist_data['High'],
                low=hist_data['Low'],
                close=hist_data['Close'],
                name='Price',
                increasing_line_color='green',
                decreasing_line_color='red'
            ))
            
            fig2.add_hrect(y0=support_level * 0.998, y1=support_level * 1.002,
                           fillcolor="green", opacity=0.15,
                           annotation_text="Support", annotation_position="left inside",
                           annotation=dict(font=dict(size=10, color="white")))
            
            fig2.add_hrect(y0=resistance_level * 0.998, y1=resistance_level * 1.002,
                           fillcolor="red", opacity=0.15,
                           annotation_text="Resistance", annotation_position="right inside",
                           annotation=dict(font=dict(size=10, color="white")))
            
            fig2.add_hline(y=spot, line_dash="dash", line_color="cyan", line_width=2,
                           annotation=dict(text=f"${spot:.2f}", font=dict(size=10, color="cyan")))
            
            fig2.update_layout(
                height=600,
                plot_bgcolor='#0e1117',
                paper_bgcolor='#0e1117',
                font=dict(color='white', size=11),
                xaxis=dict(title="Time", gridcolor='#1f1f1f', showgrid=True),
                yaxis=dict(title="Price", gridcolor='#1f1f1f', tickformat='$.2f'),
                hovermode='x unified',
                xaxis_rangeslider_visible=False,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("Unable to fetch price history")
    except:
        st.warning("Unable to fetch price history")

# Additional tabs
st.markdown("---")
tab1, tab2 = st.tabs(["📊 GEX Curves", "📋 Raw Data"])

with tab1:
    fig3 = go.Figure()
    
    fig3.add_trace(go.Scatter(x=gex_df_filtered['strike'], y=gex_df_filtered['call_gex'],
                              mode='lines', name='Call GEX', line=dict(color='green', width=3),
                              fill='tozeroy', fillcolor='rgba(0,255,0,0.2)'))
    
    fig3.add_trace(go.Scatter(x=gex_df_filtered['strike'], y=gex_df_filtered['put_gex'],
                              mode='lines', name='Put GEX', line=dict(color='red', width=3),
                              fill='tozeroy', fillcolor='rgba(255,0,0,0.2)'))
    
    fig3.add_trace(go.Scatter(x=gex_df_filtered['strike'], y=gex_df_filtered['net_gex'],
                              mode='lines', name='Net GEX', line=dict(color='gold', width=4)))
    
    fig3.add_vline(x=spot, line_dash="dash", line_color="cyan", line_width=2)
    
    fig3.update_layout(height=500, plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                      font=dict(color='white'), xaxis=dict(title="Strike", gridcolor='#1f1f1f', tickformat='$.0f'),
                      yaxis=dict(title="Gamma Exposure", gridcolor='#1f1f1f'), hovermode='x unified')
    
    st.plotly_chart(fig3, use_container_width=True)

with tab2:
    display_df = gex_df_filtered[['strike', 'call_gex', 'put_gex', 'net_gex', 'call_oi', 'put_oi', 'total_gamma']].copy()
    display_df.columns = ['Strike', 'Call GEX', 'Put GEX', 'Net GEX', 'Call OI', 'Put OI', 'Total Gamma']
    display_df = display_df.sort_values('Total Gamma', ascending=False)
    
    st.dataframe(display_df.style.format({
        'Strike': '${:.0f}', 'Call GEX': '{:,.0f}', 'Put GEX': '{:,.0f}',
        'Net GEX': '{:,.0f}', 'Call OI': '{:,.0f}', 'Put OI': '{:,.0f}',
        'Total Gamma': '{:,.0f}'
    }), use_container_width=True, height=500)

# Footer
st.markdown("---")
st.caption(f"💾 Data: Barchart.com | ⏰ {datetime.now().strftime('%H:%M:%S %Y-%m-%d')} | ⚠️ Educational purposes only")
st.caption("✨ Barchart data updates more frequently than Yahoo Finance - better OI coverage!")
