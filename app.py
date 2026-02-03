
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from math import log, sqrt
from scipy.stats import norm
import plotly.express as px

# Page config
st.set_page_config(page_title="GEX Profile Analyzer", layout="wide", initial_sidebar_state="expanded")

# -----------------------------
# Black-Scholes Greeks
# -----------------------------
def bs_greeks(S, K, T, r, sigma, option_type="call"):
    """Return delta, gamma for option."""
    if T <= 0 or sigma <= 0:
        return 0, 0
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    delta = norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * sqrt(T))
    return delta, gamma

# -----------------------------
# IV Estimation Methods
# -----------------------------
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

# -----------------------------
# GEX Calculation
# -----------------------------
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_options_data(ticker_symbol, expiry_offset=0):
    """Fetch options data and calculate GEX"""
    
    # Handle SPX - use ^SPX for price data
    price_ticker = f"^{ticker_symbol}" if ticker_symbol == "SPX" else ticker_symbol
    
    ticker = yf.Ticker(ticker_symbol)
    price_ticker_obj = yf.Ticker(price_ticker)
    
    # Get current price with better error handling
    try:
        hist = price_ticker_obj.history(period="1d", interval="1m")
        if not hist.empty:
            spot = hist["Close"].iloc[-1]
        else:
            # Fallback to daily data
            hist = price_ticker_obj.history(period="5d")
            if not hist.empty:
                spot = hist["Close"].iloc[-1]
            else:
                # Last resort - try the options ticker
                hist = ticker.history(period="5d")
                if not hist.empty:
                    spot = hist["Close"].iloc[-1]
                else:
                    return None, None, None, None, None, None, None
    except Exception as e:
        st.error(f"Could not fetch price data: {e}")
        return None, None, None, None, None, None, None
    
    # Get options
    expirations = ticker.options
    if not expirations:
        return None, None, None, None, None, None, None
    
    expiry = expirations[expiry_offset]
    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d")
    chain = ticker.option_chain(expiry)
    calls, puts = chain.calls, chain.puts
    
    # Time to expiry
    t_expiry = (expiry_dt - datetime.now()).total_seconds() / (365 * 24 * 60 * 60)
    MIN_T = 1 / (24 * 60)
    t_expiry = max(MIN_T, t_expiry)
    
    # Check IV quality and get fallback if needed
    calls_with_iv = calls[calls['impliedVolatility'] > 0]
    puts_with_iv = puts[puts['impliedVolatility'] > 0]
    iv_coverage = (len(calls_with_iv) + len(puts_with_iv)) / (len(calls) + len(puts)) * 100 if len(calls) + len(puts) > 0 else 0
    
    fallback_iv = None
    if iv_coverage < 50:
        fallback_iv = get_vix_as_iv()
        if fallback_iv is None:
            fallback_iv = estimate_iv_from_historical(price_ticker_obj, days=30)
        if fallback_iv is None:
            fallback_iv = 0.15
    
    return spot, calls, puts, t_expiry, fallback_iv, expiry, expirations

def compute_gex_density(calls, puts, spot, t_expiry, fallback_iv, r=0.05):
    """Compute GEX for each strike"""
    all_strikes = sorted(list(set(calls['strike'].tolist() + puts['strike'].tolist())))
    gex_by_strike = []
    
    for K in all_strikes:
        call_gex = 0
        put_gex = 0
        call_oi = 0
        put_oi = 0
        
        # Calls
        call_data = calls[calls['strike'] == K]
        if not call_data.empty:
            row = call_data.iloc[0]
            iv = row["impliedVolatility"]
            OI = row["openInterest"]
            call_oi = OI
            
            if iv <= 0 and fallback_iv is not None:
                iv = fallback_iv
            
            if iv > 0 and OI > 0:
                _, gamma = bs_greeks(spot, K, t_expiry, r, iv, "call")
                call_gex = gamma * OI * 100
        
        # Puts
        put_data = puts[puts['strike'] == K]
        if not put_data.empty:
            row = put_data.iloc[0]
            iv = row["impliedVolatility"]
            OI = row["openInterest"]
            put_oi = OI
            
            if iv <= 0 and fallback_iv is not None:
                iv = fallback_iv
            
            if iv > 0 and OI > 0:
                _, gamma = bs_greeks(spot, K, t_expiry, r, iv, "put")
                put_gex = gamma * OI * 100
        
        net_gex = call_gex - put_gex
        gex_by_strike.append({
            'strike': K,
            'call_gex': call_gex,
            'put_gex': put_gex,
            'net_gex': net_gex,
            'call_oi': call_oi,
            'put_oi': put_oi,
            'total_gamma': abs(call_gex) + abs(put_gex)
        })
    
    return pd.DataFrame(gex_by_strike)

# -----------------------------
# Streamlit App
# -----------------------------
st.title("🎯 GEX Profile Analyzer")
st.markdown("*Dealer gamma exposure analysis for index options*")

# Sidebar
st.sidebar.header("Settings")
ticker_symbol = st.sidebar.text_input("Ticker Symbol", "SPY").upper()
expiry_offset = st.sidebar.number_input("Expiry Index (0=nearest)", min_value=0, max_value=10, value=0)
price_range_pct = st.sidebar.slider("Strike Range (%)", min_value=1, max_value=10, value=3)
r = st.sidebar.number_input("Risk-free Rate", min_value=0.0, max_value=0.10, value=0.05, step=0.01)

# Add refresh button
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# Fetch data
with st.spinner("Fetching options data..."):
    result = fetch_options_data(ticker_symbol, expiry_offset)
    
    if result[0] is None:
        st.error(f"❌ Could not fetch options data for {ticker_symbol}")
        st.stop()
    
    spot, calls, puts, t_expiry, fallback_iv, expiry, expirations = result

# Display current info
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Current Price", f"${spot:.2f}")
with col2:
    days_to_expiry = (datetime.strptime(expiry, "%Y-%m-%d") - datetime.now()).days
    st.metric("DTE", days_to_expiry)
with col3:
    st.metric("Expiry", expiry)
with col4:
    iv_display = fallback_iv if fallback_iv else "Live IV"
    st.metric("IV Source", f"{iv_display:.2%}" if isinstance(iv_display, float) else iv_display)

# Compute GEX
gex_df = compute_gex_density(calls, puts, spot, t_expiry, fallback_iv, r)

# Filter for strikes around current price
price_range = spot * (price_range_pct / 100)
min_strike = spot - price_range
max_strike = spot + price_range
gex_df_filtered = gex_df[(gex_df['strike'] >= min_strike) & (gex_df['strike'] <= max_strike)]
gex_df_filtered = gex_df_filtered[(gex_df_filtered['call_gex'] != 0) | (gex_df_filtered['put_gex'] != 0)]

if len(gex_df_filtered) == 0:
    st.warning("⚠️ No valid GEX data in selected range")
    st.stop()

# Calculate summary metrics
total_call_gex = gex_df_filtered['call_gex'].sum()
total_put_gex = gex_df_filtered['put_gex'].sum()
net_gex = total_call_gex - total_put_gex

# Display summary
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Call GEX", f"{total_call_gex:,.0f}", delta="Dealers short gamma")
with col2:
    st.metric("Total Put GEX", f"{total_put_gex:,.0f}", delta="Dealers long gamma")
with col3:
    gex_condition = "Positive (Dampening)" if net_gex > 0 else "Negative (Accelerating)"
    st.metric("Net GEX", f"{net_gex:,.0f}", delta=gex_condition)

# Main visualization
st.markdown("---")
st.subheader("Gamma Exposure Profile")

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["📊 Horizontal Profile", "📈 Line Chart", "📋 Data Table"])

with tab1:
    # Horizontal bar chart (like the reference image)
    fig = go.Figure()
    
    # Add call gamma (positive, green)
    fig.add_trace(go.Bar(
        y=gex_df_filtered['strike'],
        x=gex_df_filtered['call_gex'],
        orientation='h',
        name='Call GEX (Dealers Short)',
        marker_color='lightgreen',
        hovertemplate='Strike: $%{y:.0f}<br>Call GEX: %{x:,.0f}<extra></extra>'
    ))
    
    # Add put gamma (negative, red)
    fig.add_trace(go.Bar(
        y=gex_df_filtered['strike'],
        x=-gex_df_filtered['put_gex'],  # Negative for left side
        orientation='h',
        name='Put GEX (Dealers Long)',
        marker_color='lightcoral',
        hovertemplate='Strike: $%{y:.0f}<br>Put GEX: %{x:,.0f}<extra></extra>'
    ))
    
    # Add current price line
    fig.add_hline(y=spot, line_dash="dash", line_color="blue", 
                  annotation_text=f"Spot: ${spot:.2f}", 
                  annotation_position="right")
    
    # Find max GEX strike (zero gamma point)
    max_gex_idx = gex_df_filtered['net_gex'].abs().idxmax()
    max_gex_strike = gex_df_filtered.loc[max_gex_idx, 'strike']
    fig.add_hline(y=max_gex_strike, line_dash="dot", line_color="gold",
                  annotation_text=f"Zero Gamma: ${max_gex_strike:.0f}",
                  annotation_position="left")
    
    fig.update_layout(
        barmode='overlay',
        height=600,
        xaxis_title="Gamma Exposure",
        yaxis_title="Strike Price",
        hovermode='closest',
        showlegend=True,
        yaxis=dict(tickformat='$.0f')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation
    st.markdown("### 📖 Interpretation")
    if net_gex > 0:
        st.success("**Positive Net GEX**: Dealers are net short gamma. Market likely to see dampened volatility as dealers hedge by buying dips and selling rips.")
    else:
        st.warning("**Negative Net GEX**: Dealers are net long gamma. Market likely to see amplified moves as dealers hedge by selling dips and buying rips.")

with tab2:
    # Line chart view
    fig2 = go.Figure()
    
    fig2.add_trace(go.Scatter(
        x=gex_df_filtered['strike'],
        y=gex_df_filtered['call_gex'],
        mode='lines+markers',
        name='Call GEX',
        line=dict(color='green', width=3)
    ))
    
    fig2.add_trace(go.Scatter(
        x=gex_df_filtered['strike'],
        y=gex_df_filtered['put_gex'],
        mode='lines+markers',
        name='Put GEX',
        line=dict(color='red', width=3)
    ))
    
    fig2.add_trace(go.Scatter(
        x=gex_df_filtered['strike'],
        y=gex_df_filtered['net_gex'],
        mode='lines+markers',
        name='Net GEX',
        line=dict(color='gold', width=4)
    ))
    
    fig2.add_vline(x=spot, line_dash="dash", line_color="blue",
                   annotation_text=f"Spot: ${spot:.2f}")
    
    fig2.update_layout(
        height=500,
        xaxis_title="Strike Price",
        yaxis_title="Gamma Exposure",
        hovermode='x unified',
        xaxis=dict(tickformat='$.0f')
    )
    
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    # Data table
    display_df = gex_df_filtered[['strike', 'call_gex', 'put_gex', 'net_gex', 'call_oi', 'put_oi', 'total_gamma']].copy()
    display_df.columns = ['Strike', 'Call GEX', 'Put GEX', 'Net GEX', 'Call OI', 'Put OI', 'Total Gamma']
    display_df = display_df.sort_values('Total Gamma', ascending=False)
    
    st.dataframe(
        display_df.style.format({
            'Strike': '${:.0f}',
            'Call GEX': '{:,.0f}',
            'Put GEX': '{:,.0f}',
            'Net GEX': '{:,.0f}',
            'Call OI': '{:,.0f}',
            'Put OI': '{:,.0f}',
            'Total Gamma': '{:,.0f}'
        }),
        use_container_width=True,
        height=400
    )

# Add price history chart with gamma zones
st.markdown("---")
st.subheader("Price Action with Gamma Zones")

# Fetch historical data - use correct ticker for price data
price_ticker_for_chart = f"^{ticker_symbol}" if ticker_symbol == "SPX" else ticker_symbol
hist_data = yf.Ticker(price_ticker_for_chart).history(period="5d", interval="5m")

if not hist_data.empty:
    # Identify key gamma levels
    support_level = gex_df_filtered.loc[gex_df_filtered['put_gex'].idxmax(), 'strike']
    resistance_level = gex_df_filtered.loc[gex_df_filtered['call_gex'].idxmax(), 'strike']
    
    fig3 = go.Figure()
    
    # Price line
    fig3.add_trace(go.Scatter(
        x=hist_data.index,
        y=hist_data['Close'],
        mode='lines',
        name='Price',
        line=dict(color='white', width=2)
    ))
    
    # Add gamma zones
    fig3.add_hrect(y0=support_level * 0.995, y1=support_level * 1.005,
                   fillcolor="green", opacity=0.2,
                   annotation_text="Support Zone", annotation_position="left")
    
    fig3.add_hrect(y0=resistance_level * 0.995, y1=resistance_level * 1.005,
                   fillcolor="red", opacity=0.2,
                   annotation_text="Resistance Zone", annotation_position="right")
    
    fig3.update_layout(
        height=400,
        xaxis_title="Time",
        yaxis_title="Price",
        hovermode='x unified',
        template='plotly_dark'
    )
    
    st.plotly_chart(fig3, use_container_width=True)

# Footer
st.markdown("---")
st.caption(f"Data source: Yahoo Finance | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption("⚠️ For educational purposes only. Not financial advice.")
