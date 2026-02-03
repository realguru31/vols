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
st.title("🎯 GEX Profile Analyzer (DIAGNOSTIC MODE)")
st.markdown("*Professional dealer gamma exposure analysis with data diagnostics*")

# Sidebar
st.sidebar.header("⚙️ Settings")
TICKER = st.sidebar.text_input("Ticker Symbol", "SPY").upper()
EXPIRY_OFFSET = st.sidebar.number_input("Expiry Index (0=nearest)", min_value=0, max_value=10, value=0, step=1)
price_range_pct = st.sidebar.slider("Strike Range (%)", min_value=1, max_value=20, value=10)
r = st.sidebar.number_input("Risk-free Rate", min_value=0.0, max_value=0.10, value=0.05, step=0.01)

if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

# Constants
MIN_T = 1 / (24 * 60)

# Fetch data with DIAGNOSTICS
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
        
        st.success(f"✅ Got spot price: ${spot:.2f}")
        
        # Get options
        expirations = options_ticker.options
        if not expirations:
            st.error(f"❌ No options expirations available for {TICKER}")
            st.stop()
        
        st.success(f"✅ Found {len(expirations)} expirations available")
        
        # Show all available expirations
        with st.expander("📅 Available Expirations (click to expand)"):
            for i, exp in enumerate(expirations[:20]):  # Show first 20
                exp_dt = datetime.strptime(exp, "%Y-%m-%d")
                dte = (exp_dt - datetime.now()).days
                st.write(f"Index {i}: {exp} (DTE: {dte})")
        
        expiry = expirations[EXPIRY_OFFSET]
        expiry_dt = datetime.strptime(expiry, "%Y-%m-%d")
        chain = options_ticker.option_chain(expiry)
        calls, puts = chain.calls, chain.puts
        
        st.success(f"✅ Got options chain: {len(calls)} calls, {len(puts)} puts")
        
        # DIAGNOSTIC: Show sample of actual data
        st.subheader("🔬 Data Quality Diagnostics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 CALLS Sample (first 5 rows)**")
            if len(calls) > 0:
                sample_calls = calls[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']].head()
                st.dataframe(sample_calls, use_container_width=True)
            
            # Count valid IV
            calls_with_iv = calls[calls['impliedVolatility'] > 0]
            calls_with_oi = calls[calls['openInterest'] > 0]
            st.metric("Calls with IV > 0", f"{len(calls_with_iv)} / {len(calls)}")
            st.metric("Calls with OI > 0", f"{len(calls_with_oi)} / {len(calls)}")
            
        with col2:
            st.markdown("**📊 PUTS Sample (first 5 rows)**")
            if len(puts) > 0:
                sample_puts = puts[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']].head()
                st.dataframe(sample_puts, use_container_width=True)
            
            # Count valid IV
            puts_with_iv = puts[puts['impliedVolatility'] > 0]
            puts_with_oi = puts[puts['openInterest'] > 0]
            st.metric("Puts with IV > 0", f"{len(puts_with_iv)} / {len(puts)}")
            st.metric("Puts with OI > 0", f"{len(puts_with_oi)} / {len(puts)}")
        
        # Time to expiry
        t_expiry = (expiry_dt - datetime.now()).total_seconds() / (365 * 24 * 60 * 60)
        days_to_expiry = (expiry_dt - datetime.now()).days
        hours_to_expiry = (expiry_dt - datetime.now()).total_seconds() / 3600
        
        st.info(f"⏰ Time to expiry: {days_to_expiry} days ({hours_to_expiry:.1f} hours) - T={t_expiry:.6f}")
        
        if expiry_dt.date() == datetime.now().date():
            expiry_label = "0DTE"
            t_expiry = max(MIN_T, t_expiry)
        else:
            expiry_label = f"{t_expiry*365:.1f} days"
            t_expiry = max(MIN_T, t_expiry)
        
        # Check IV quality
        total_options = len(calls) + len(puts)
        total_with_iv = len(calls_with_iv) + len(puts_with_iv)
        iv_coverage = (total_with_iv / total_options * 100) if total_options > 0 else 0
        
        st.warning(f"📊 IV Coverage: {iv_coverage:.1f}% ({total_with_iv}/{total_options} options have IV > 0)")
        
        # Choose IV strategy
        fallback_iv = None
        if iv_coverage < 50:
            st.warning("⚠️ Low IV coverage - using fallback methods")
            fallback_iv = get_vix_as_iv()
            if fallback_iv is None:
                fallback_iv = estimate_iv_from_historical(price_ticker, days=30)
            if fallback_iv is None:
                fallback_iv = 0.15
            st.info(f"📈 Using fallback IV: {fallback_iv:.4f}")
        else:
            st.success(f"✅ Good IV coverage ({iv_coverage:.1f}%)")
        
    except Exception as e:
        st.error(f"❌ Error fetching data: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

st.markdown("---")

# Display current info
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Spot", f"${spot:.2f}")
with col2:
    st.metric("Ticker", TICKER)
with col3:
    st.metric("DTE", f"{days_to_expiry}d")
with col4:
    st.metric("Expiry", expiry.split('-')[1] + '/' + expiry.split('-')[2])
with col5:
    if fallback_iv:
        st.metric("IV", f"{fallback_iv:.1%}*")
    else:
        st.metric("IV", "Live")

st.markdown("---")

# Compute GEX density with DIAGNOSTICS
st.subheader("⚙️ Computing GEX...")

def compute_gex_density():
    all_strikes = sorted(list(set(calls['strike'].tolist() + puts['strike'].tolist())))
    gex_by_strike = []
    
    strikes_with_call_gex = 0
    strikes_with_put_gex = 0
    
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
                if call_gex > 0:
                    strikes_with_call_gex += 1
        
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
                if put_gex > 0:
                    strikes_with_put_gex += 1
        
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
    
    st.info(f"✅ Strikes with Call GEX > 0: {strikes_with_call_gex}")
    st.info(f"✅ Strikes with Put GEX > 0: {strikes_with_put_gex}")
    
    return pd.DataFrame(gex_by_strike)

gex_df = compute_gex_density()

st.success(f"✅ Computed GEX for {len(gex_df)} total strikes")

# Show GEX statistics
if len(gex_df) > 0:
    non_zero_gex = gex_df[(gex_df['call_gex'] != 0) | (gex_df['put_gex'] != 0)]
    st.info(f"📊 Strikes with non-zero GEX: {len(non_zero_gex)} / {len(gex_df)}")
    
    if len(non_zero_gex) > 0:
        st.info(f"📊 Strike range with GEX: ${non_zero_gex['strike'].min():.0f} - ${non_zero_gex['strike'].max():.0f}")
        st.info(f"📊 Current spot: ${spot:.2f}")

# Filter for strikes around current price
price_range = spot * (price_range_pct / 100)
min_strike = spot - price_range
max_strike = spot + price_range

st.info(f"🔍 Filtering for strikes in range: ${min_strike:.0f} - ${max_strike:.0f}")

gex_df_filtered = gex_df[(gex_df['strike'] >= min_strike) & (gex_df['strike'] <= max_strike)].copy()

st.info(f"📊 Strikes in price range: {len(gex_df_filtered)}")

# Remove zero GEX strikes
gex_df_filtered = gex_df_filtered[(gex_df_filtered['call_gex'] != 0) | (gex_df_filtered['put_gex'] != 0)]

st.info(f"📊 Non-zero GEX strikes in range: {len(gex_df_filtered)}")

# Show what we have even if empty
if len(gex_df_filtered) > 0:
    st.success(f"✅ Ready to plot {len(gex_df_filtered)} strikes!")
    
    # Show sample of filtered data
    with st.expander("📊 Sample of GEX data (click to expand)"):
        display_sample = gex_df_filtered[['strike', 'call_gex', 'put_gex', 'net_gex', 'call_oi', 'put_oi']].head(10)
        st.dataframe(display_sample, use_container_width=True)
else:
    st.error("❌ No valid GEX data in selected range!")
    st.warning("🔍 Diagnostics:")
    st.write(f"- Total strikes computed: {len(gex_df)}")
    st.write(f"- Strikes with any GEX: {len(non_zero_gex) if len(gex_df) > 0 else 0}")
    st.write(f"- Your selected range: ${min_strike:.0f} - ${max_strike:.0f}")
    st.write(f"- Current spot: ${spot:.2f}")
    st.warning("💡 Try:")
    st.write("1. Increase Strike Range % slider to 15-20%")
    st.write("2. Change Expiry Index to next expiration")
    st.write("3. Check if market is open (options need active trading)")
    st.stop()

st.markdown("---")

# Calculate summary metrics
total_call_gex = gex_df_filtered['call_gex'].sum()
total_put_gex = gex_df_filtered['put_gex'].sum()
net_gex = total_call_gex - total_put_gex

# Display GEX summary
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Call GEX", f"{total_call_gex:,.0f}")
with col2:
    st.metric("Put GEX", f"{total_put_gex:,.0f}")
with col3:
    gex_sign = "🟢 +" if net_gex > 0 else "🔴 -"
    st.metric("Net GEX", f"{gex_sign}{abs(net_gex):,.0f}")

# Market regime
if net_gex > 0:
    st.success("**📊 POSITIVE GEX:** Dampening regime - lower volatility expected")
else:
    st.error("**⚡ NEGATIVE GEX:** Amplifying regime - higher volatility expected")

st.markdown("---")
st.subheader("📊 Gamma Exposure Profile")

# Create professional horizontal profile
fig = go.Figure()

fig.add_trace(go.Bar(
    y=gex_df_filtered['strike'],
    x=gex_df_filtered['call_gex'],
    orientation='h',
    name='Call GEX',
    marker=dict(color='rgba(0, 255, 0, 0.6)', line=dict(color='rgba(0, 200, 0, 1.0)', width=1)),
    hovertemplate='<b>$%{y:.0f}</b><br>Call GEX: %{x:,.0f}<extra></extra>'
))

fig.add_trace(go.Bar(
    y=gex_df_filtered['strike'],
    x=-gex_df_filtered['put_gex'],
    orientation='h',
    name='Put GEX',
    marker=dict(color='rgba(255, 0, 0, 0.6)', line=dict(color='rgba(200, 0, 0, 1.0)', width=1)),
    hovertemplate='<b>$%{y:.0f}</b><br>Put GEX: %{x:,.0f}<extra></extra>'
))

fig.add_hline(y=spot, line_dash="dash", line_color="cyan", line_width=3,
              annotation=dict(text=f"SPOT: ${spot:.2f}", font=dict(size=14, color="cyan"), bgcolor="rgba(0,0,0,0.8)"),
              annotation_position="right")

if len(gex_df_filtered) > 0:
    max_gex_idx = gex_df_filtered['net_gex'].abs().idxmax()
    max_gex_strike = gex_df_filtered.loc[max_gex_idx, 'strike']
    fig.add_hline(y=max_gex_strike, line_dash="dot", line_color="gold", line_width=2,
                  annotation=dict(text=f"MAX GEX: ${max_gex_strike:.0f}", font=dict(size=12, color="gold"), bgcolor="rgba(0,0,0,0.7)"),
                  annotation_position="left")

fig.update_layout(
    barmode='overlay',
    height=700,
    plot_bgcolor='#0e1117',
    paper_bgcolor='#0e1117',
    font=dict(color='white', size=12),
    xaxis=dict(title="Gamma Exposure", gridcolor='#1f1f1f', showgrid=True, tickformat=','),
    yaxis=dict(title="Strike Price", gridcolor='#1f1f1f', tickformat='$.0f'),
    hovermode='closest',
    showlegend=True,
    legend=dict(bgcolor='rgba(0,0,0,0.5)', bordercolor='white', borderwidth=1)
)

st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.caption(f"💾 Yahoo Finance | ⏰ {datetime.now().strftime('%H:%M:%S %Y-%m-%d')} | ⚠️ Educational only")
