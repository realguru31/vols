import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from math import log, sqrt
from scipy.stats import norm

# Page config
st.set_page_config(page_title="GEX Profile Analyzer", layout="wide", initial_sidebar_state="expanded")

# Black-Scholes Greeks
def bs_greeks(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0:
        return 0, 0
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    delta = norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * sqrt(T))
    return delta, gamma

# IV Estimation Methods
def estimate_iv_from_historical(ticker_obj, days=30):
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
    try:
        vix = yf.Ticker("^VIX")
        vix_value = vix.history(period="1d")['Close'].iloc[-1]
        iv_estimate = vix_value / 100
        return iv_estimate
    except:
        return None

# Streamlit App
st.title("🎯 GEX Profile Analyzer")
st.markdown("*Professional dealer gamma exposure analysis*")

# Sidebar
st.sidebar.header("⚙️ Settings")
TICKER = st.sidebar.text_input("Ticker Symbol", "SPY").upper()
EXPIRY_OFFSET = st.sidebar.number_input("Expiry Index (0=nearest)", min_value=0, max_value=10, value=0, step=1)
price_range_pct = st.sidebar.slider("Strike Range (%)", min_value=1, max_value=10, value=3)
r = st.sidebar.number_input("Risk-free Rate", min_value=0.0, max_value=0.10, value=0.05, step=0.01)

if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 How to Read")
st.sidebar.markdown("""
**Green bars (right):** Call GEX  
Dealers are **short gamma** here

**Red bars (left):** Put GEX  
Dealers are **long gamma** here

**Blue line:** Current spot price

**Gold line:** Max GEX level  
(Zero gamma / pin point)
""")

# Constants
MIN_T = 1 / (24 * 60)

# Fetch data
with st.spinner("🔍 Fetching options data..."):
    try:
        # Handle SPX - use ^SPX for price
        if TICKER == "SPX":
            price_ticker = yf.Ticker("^SPX")
            options_ticker = yf.Ticker("SPX")
        else:
            price_ticker = yf.Ticker(TICKER)
            options_ticker = price_ticker
        
        # Get current price
        try:
            spot = price_ticker.history(period="1d", interval="1m")["Close"].iloc[-1]
        except:
            spot = price_ticker.history(period="5d")["Close"].iloc[-1]
        
        # Get options
        expirations = options_ticker.options
        if not expirations:
            st.error(f"❌ No options expirations available for {TICKER}")
            st.stop()
        
        expiry = expirations[EXPIRY_OFFSET]
        expiry_dt = datetime.strptime(expiry, "%Y-%m-%d")
        chain = options_ticker.option_chain(expiry)
        calls, puts = chain.calls, chain.puts
        
        # Time to expiry
        t_expiry = (expiry_dt - datetime.now()).total_seconds() / (365 * 24 * 60 * 60)
        if expiry_dt.date() == datetime.now().date():
            expiry_label = "0DTE"
            t_expiry = max(MIN_T, t_expiry)
        else:
            expiry_label = f"{t_expiry*365:.1f} days"
            t_expiry = max(MIN_T, t_expiry)
        
        # Check IV quality
        calls_with_iv = calls[calls['impliedVolatility'] > 0]
        puts_with_iv = puts[puts['impliedVolatility'] > 0]
        iv_coverage = (len(calls_with_iv) + len(puts_with_iv)) / (len(calls) + len(puts)) * 100
        
        # Choose IV strategy
        fallback_iv = None
        if iv_coverage < 50:
            fallback_iv = get_vix_as_iv()
            if fallback_iv is None:
                fallback_iv = estimate_iv_from_historical(price_ticker, days=30)
            if fallback_iv is None:
                fallback_iv = 0.15
        
    except Exception as e:
        st.error(f"❌ Error fetching data: {str(e)}")
        st.stop()

# Display current info - compact professional style
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Spot", f"${spot:.2f}")
with col2:
    st.metric("Ticker", TICKER)
with col3:
    days_to_expiry = (expiry_dt - datetime.now()).days
    st.metric("DTE", f"{days_to_expiry}d")
with col4:
    st.metric("Expiry", expiry.split('-')[1] + '/' + expiry.split('-')[2])
with col5:
    if fallback_iv:
        st.metric("IV", f"{fallback_iv:.1%}*")
    else:
        st.metric("IV", "Live")

# Compute GEX density (exact same logic as matplotlib script)
def compute_gex_density():
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

gex_df = compute_gex_density()

# Filter for strikes around current price (exact same as matplotlib script)
price_range = spot * (price_range_pct / 100)
min_strike = spot - price_range
max_strike = spot + price_range
gex_df_filtered = gex_df[(gex_df['strike'] >= min_strike) & (gex_df['strike'] <= max_strike)].copy()

# Remove zero GEX strikes (exact same as matplotlib script)
gex_df_filtered = gex_df_filtered[(gex_df_filtered['call_gex'] != 0) | (gex_df_filtered['put_gex'] != 0)]

if len(gex_df_filtered) == 0:
    st.error("⚠️ No valid GEX data in selected range")
    st.info(f"💡 Computed {len(gex_df)} total strikes, but none in ${min_strike:.0f}-${max_strike:.0f} range with non-zero GEX")
    st.info("Try increasing Strike Range % in sidebar")
    st.stop()

# Calculate summary metrics
total_call_gex = gex_df_filtered['call_gex'].sum()
total_put_gex = gex_df_filtered['put_gex'].sum()
net_gex = total_call_gex - total_put_gex

# Display GEX summary - professional compact style
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Call GEX", f"{total_call_gex:,.0f}", help="Dealers SHORT gamma at these strikes")
with col2:
    st.metric("Put GEX", f"{total_put_gex:,.0f}", help="Dealers LONG gamma at these strikes")
with col3:
    gex_sign = "🟢 +" if net_gex > 0 else "🔴 -"
    st.metric("Net GEX", f"{gex_sign}{abs(net_gex):,.0f}")

# Market regime interpretation
if net_gex > 0:
    st.success("**📊 POSITIVE GEX REGIME:** Market pressure is dampening. Dealers hedge by buying dips and selling rips → Lower volatility expected")
else:
    st.error("**⚡ NEGATIVE GEX REGIME:** Market pressure is amplifying. Dealers hedge by selling dips and buying rips → Higher volatility expected")

st.markdown("---")

# MAIN PROFESSIONAL HORIZONTAL PROFILE (like reference image)
st.subheader("📊 Gamma Exposure Profile")

# Create the professional horizontal bar chart
fig = go.Figure()

# Call GEX - positive bars going right (green like reference)
fig.add_trace(go.Bar(
    y=gex_df_filtered['strike'],
    x=gex_df_filtered['call_gex'],
    orientation='h',
    name='Call GEX (Dealers Short Gamma)',
    marker=dict(
        color='rgba(0, 255, 0, 0.6)',
        line=dict(color='rgba(0, 200, 0, 1.0)', width=1)
    ),
    hovertemplate='<b>Strike: $%{y:.0f}</b><br>Call GEX: %{x:,.0f}<br>Dealers SHORT gamma<extra></extra>'
))

# Put GEX - negative bars going left (red like reference)
fig.add_trace(go.Bar(
    y=gex_df_filtered['strike'],
    x=-gex_df_filtered['put_gex'],  # Negative for left side
    orientation='h',
    name='Put GEX (Dealers Long Gamma)',
    marker=dict(
        color='rgba(255, 0, 0, 0.6)',
        line=dict(color='rgba(200, 0, 0, 1.0)', width=1)
    ),
    hovertemplate='<b>Strike: $%{y:.0f}</b><br>Put GEX: %{x:,.0f}<br>Dealers LONG gamma<extra></extra>'
))

# Current spot price - blue dashed line (like reference)
fig.add_hline(
    y=spot, 
    line_dash="dash", 
    line_color="cyan", 
    line_width=3,
    annotation=dict(
        text=f"SPOT: ${spot:.2f}",
        font=dict(size=14, color="cyan"),
        bgcolor="rgba(0,0,0,0.8)"
    ),
    annotation_position="right"
)

# Zero gamma point / max GEX strike (gold like reference)
if len(gex_df_filtered) > 0:
    max_gex_idx = gex_df_filtered['net_gex'].abs().idxmax()
    max_gex_strike = gex_df_filtered.loc[max_gex_idx, 'strike']
    fig.add_hline(
        y=max_gex_strike, 
        line_dash="dot", 
        line_color="gold",
        line_width=2,
        annotation=dict(
            text=f"ZERO Γ: ${max_gex_strike:.0f}",
            font=dict(size=12, color="gold"),
            bgcolor="rgba(0,0,0,0.7)"
        ),
        annotation_position="left"
    )

# Professional dark theme styling (like reference image)
fig.update_layout(
    barmode='overlay',
    height=700,
    plot_bgcolor='#0e1117',
    paper_bgcolor='#0e1117',
    font=dict(color='white', size=12),
    xaxis=dict(
        title="Gamma Exposure",
        gridcolor='#1f1f1f',
        zerolinecolor='#1f1f1f',
        showgrid=True,
        tickformat=',',
        title_font=dict(size=14)
    ),
    yaxis=dict(
        title="Strike Price",
        gridcolor='#1f1f1f',
        tickformat='$.0f',
        title_font=dict(size=14)
    ),
    hovermode='closest',
    showlegend=True,
    legend=dict(
        bgcolor='rgba(0,0,0,0.5)',
        bordercolor='white',
        borderwidth=1,
        font=dict(size=11)
    )
)

st.plotly_chart(fig, use_container_width=True)

# Additional tabs for other views
st.markdown("---")
tab1, tab2 = st.tabs(["📈 Alternative Views", "📋 Raw Data"])

with tab1:
    # Line chart view
    fig2 = go.Figure()
    
    fig2.add_trace(go.Scatter(
        x=gex_df_filtered['strike'],
        y=gex_df_filtered['call_gex'],
        mode='lines+markers',
        name='Call GEX',
        line=dict(color='green', width=3),
        marker=dict(size=6)
    ))
    
    fig2.add_trace(go.Scatter(
        x=gex_df_filtered['strike'],
        y=gex_df_filtered['put_gex'],
        mode='lines+markers',
        name='Put GEX',
        line=dict(color='red', width=3),
        marker=dict(size=6)
    ))
    
    fig2.add_trace(go.Scatter(
        x=gex_df_filtered['strike'],
        y=gex_df_filtered['net_gex'],
        mode='lines+markers',
        name='Net GEX',
        line=dict(color='gold', width=4),
        marker=dict(size=8)
    ))
    
    fig2.add_vline(x=spot, line_dash="dash", line_color="cyan",
                   annotation_text=f"Spot: ${spot:.2f}")
    
    fig2.update_layout(
        height=500,
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='white'),
        xaxis=dict(title="Strike Price", gridcolor='#1f1f1f', tickformat='$.0f'),
        yaxis=dict(title="Gamma Exposure", gridcolor='#1f1f1f'),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
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
        height=500
    )

# Footer
st.markdown("---")
st.caption(f"💾 Data: Yahoo Finance | ⏰ Updated: {datetime.now().strftime('%H:%M:%S %Y-%m-%d')} | ⚠️ Educational purposes only")
if fallback_iv:
    st.caption(f"📊 Using estimated IV ({fallback_iv:.2%}) - for live IV run during market hours")
