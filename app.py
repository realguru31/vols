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

# IV Estimation
def estimate_iv_from_historical(ticker_obj, days=30):
    try:
        hist = ticker_obj.history(period=f"{days}d")
        if len(hist) < 2:
            return None
        returns = np.log(hist['Close'] / hist['Close'].shift(1))
        return returns.std() * np.sqrt(252)
    except:
        return None

def get_vix_as_iv():
    try:
        vix = yf.Ticker("^VIX")
        vix_value = vix.history(period="1d")['Close'].iloc[-1]
        return vix_value / 100
    except:
        return None

# App
st.title("🎯 GEX Profile Analyzer")
st.markdown("*Professional dealer gamma exposure analysis*")

# Sidebar
st.sidebar.header("⚙️ Settings")
TICKER = st.sidebar.text_input("Ticker Symbol", "SPY").upper()
EXPIRY_OFFSET = st.sidebar.number_input("Expiry Index (0=nearest)", min_value=0, max_value=10, value=0, step=1)
price_range_pct = st.sidebar.slider("Strike Range (%)", min_value=1, max_value=20, value=10)
r = st.sidebar.number_input("Risk-free Rate", min_value=0.0, max_value=0.10, value=0.05, step=0.01)

if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("⏰ **Note:** Open Interest data updates during market hours (9:30 AM - 4 PM ET). Outside these hours, use longer-dated expiries (7+ days).")

MIN_T = 1 / (24 * 60)

# Fetch data
with st.spinner("🔍 Fetching options data..."):
    try:
        if TICKER == "SPX":
            price_ticker = yf.Ticker("^SPX")
            options_ticker = yf.Ticker("SPX")
        else:
            price_ticker = yf.Ticker(TICKER)
            options_ticker = price_ticker
        
        # Get spot
        try:
            spot = price_ticker.history(period="1d", interval="1m")["Close"].iloc[-1]
        except:
            spot = price_ticker.history(period="5d")["Close"].iloc[-1]
        
        # Get options
        expirations = options_ticker.options
        if not expirations:
            st.error(f"❌ No options available for {TICKER}")
            st.stop()
        
        expiry = expirations[EXPIRY_OFFSET]
        expiry_dt = datetime.strptime(expiry, "%Y-%m-%d")
        chain = options_ticker.option_chain(expiry)
        calls, puts = chain.calls, chain.puts
        
        # Time calculations
        t_expiry = (expiry_dt - datetime.now()).total_seconds() / (365 * 24 * 60 * 60)
        days_to_expiry = (expiry_dt - datetime.now()).days
        
        if expiry_dt.date() == datetime.now().date():
            expiry_label = "0DTE"
            t_expiry = max(MIN_T, t_expiry)
        else:
            expiry_label = f"{t_expiry*365:.1f}d"
            t_expiry = max(MIN_T, t_expiry)
        
        # Check data quality
        calls_with_iv = calls[calls['impliedVolatility'] > 0]
        puts_with_iv = puts[puts['impliedVolatility'] > 0]
        calls_with_oi = calls[calls['openInterest'] > 0]
        puts_with_oi = puts[puts['openInterest'] > 0]
        
        iv_coverage = (len(calls_with_iv) + len(puts_with_iv)) / (len(calls) + len(puts)) * 100
        oi_coverage = (len(calls_with_oi) + len(puts_with_oi)) / (len(calls) + len(puts)) * 100
        
        # Fallback IV
        fallback_iv = None
        if iv_coverage < 50:
            fallback_iv = get_vix_as_iv()
            if fallback_iv is None:
                fallback_iv = estimate_iv_from_historical(price_ticker, days=30)
            if fallback_iv is None:
                fallback_iv = 0.15
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.stop()

# Header metrics
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
    st.metric("IV", f"{fallback_iv:.1%}*" if fallback_iv else "Live")

# OI Warning
if oi_coverage < 5:
    st.error("⚠️ **ZERO OPEN INTEREST DETECTED**")
    st.warning(f"""
    **Why:** Yahoo Finance doesn't update OI outside market hours for short-dated options.
    
    **Solutions:**
    1. ⏰ Run during market hours (9:30 AM - 4 PM ET)
    2. 📅 Change Expiry Index to 1 or 2 (longer-dated options)
    3. 🔄 Try a different ticker
    
    **Current Coverage:** {oi_coverage:.1f}% options have OI > 0
    """)
    
    if st.button("🔄 Try Next Expiry (Index = 1)"):
        st.session_state.expiry_offset = 1
        st.rerun()
    
    st.info("💡 The app will work perfectly once OI data is available!")
    st.stop()

# Compute GEX
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

# Filter strikes
price_range = spot * (price_range_pct / 100)
min_strike = spot - price_range
max_strike = spot + price_range
gex_df_filtered = gex_df[(gex_df['strike'] >= min_strike) & (gex_df['strike'] <= max_strike)].copy()
gex_df_filtered = gex_df_filtered[(gex_df_filtered['call_gex'] != 0) | (gex_df_filtered['put_gex'] != 0)]

if len(gex_df_filtered) == 0:
    st.error("⚠️ No GEX data in selected range")
    st.info(f"Computed {len(gex_df)} strikes, but none in ${min_strike:.0f}-${max_strike:.0f} with GEX")
    st.info("💡 Try increasing Strike Range % slider")
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
    st.success("**📊 POSITIVE GEX:** Dampening regime - dealers hedge by buying dips/selling rips → Lower volatility")
else:
    st.error("**⚡ NEGATIVE GEX:** Amplifying regime - dealers hedge by selling dips/buying rips → Higher volatility")

st.markdown("---")

# Main visualization
col_chart1, col_chart2 = st.columns([1, 1])

with col_chart1:
    st.subheader("📊 Gamma Profile by Strike")
    
    # Horizontal bar chart
    fig1 = go.Figure()
    
    fig1.add_trace(go.Bar(
        y=gex_df_filtered['strike'],
        x=gex_df_filtered['call_gex'],
        orientation='h',
        name='Call GEX',
        marker=dict(color='rgba(0, 255, 0, 0.6)', line=dict(color='rgba(0, 200, 0, 1)', width=1)),
        hovertemplate='<b>$%{y:.0f}</b><br>Call GEX: %{x:,.0f}<extra></extra>'
    ))
    
    fig1.add_trace(go.Bar(
        y=gex_df_filtered['strike'],
        x=-gex_df_filtered['put_gex'],
        orientation='h',
        name='Put GEX',
        marker=dict(color='rgba(255, 0, 0, 0.6)', line=dict(color='rgba(200, 0, 0, 1)', width=1)),
        hovertemplate='<b>$%{y:.0f}</b><br>Put GEX: %{x:,.0f}<extra></extra>'
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
    
    # Get historical price data
    price_ticker_chart = f"^{TICKER}" if TICKER == "SPX" else TICKER
    hist_data = yf.Ticker(price_ticker_chart).history(period="5d", interval="5m")
    
    if not hist_data.empty:
        # Find key gamma levels
        support_level = gex_df_filtered.loc[gex_df_filtered['put_gex'].idxmax(), 'strike']
        resistance_level = gex_df_filtered.loc[gex_df_filtered['call_gex'].idxmax(), 'strike']
        
        fig2 = go.Figure()
        
        # Candlestick chart
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
        
        # Add gamma zones
        fig2.add_hrect(
            y0=support_level * 0.998, y1=support_level * 1.002,
            fillcolor="green", opacity=0.15,
            annotation_text="Support", annotation_position="left inside",
            annotation=dict(font=dict(size=10, color="white"))
        )
        
        fig2.add_hrect(
            y0=resistance_level * 0.998, y1=resistance_level * 1.002,
            fillcolor="red", opacity=0.15,
            annotation_text="Resistance", annotation_position="right inside",
            annotation=dict(font=dict(size=10, color="white"))
        )
        
        # Current spot line
        fig2.add_hline(y=spot, line_dash="dash", line_color="cyan", line_width=2,
                       annotation=dict(text=f"Spot: ${spot:.2f}", font=dict(size=10, color="cyan")))
        
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
        st.warning("Unable to fetch historical price data")

st.markdown("---")

# Additional tabs
tab1, tab2 = st.tabs(["📊 GEX Curves", "📋 Raw Data"])

with tab1:
    fig3 = go.Figure()
    
    fig3.add_trace(go.Scatter(
        x=gex_df_filtered['strike'],
        y=gex_df_filtered['call_gex'],
        mode='lines',
        name='Call GEX',
        line=dict(color='green', width=3),
        fill='tozeroy',
        fillcolor='rgba(0,255,0,0.2)'
    ))
    
    fig3.add_trace(go.Scatter(
        x=gex_df_filtered['strike'],
        y=gex_df_filtered['put_gex'],
        mode='lines',
        name='Put GEX',
        line=dict(color='red', width=3),
        fill='tozeroy',
        fillcolor='rgba(255,0,0,0.2)'
    ))
    
    fig3.add_trace(go.Scatter(
        x=gex_df_filtered['strike'],
        y=gex_df_filtered['net_gex'],
        mode='lines',
        name='Net GEX',
        line=dict(color='gold', width=4)
    ))
    
    fig3.add_vline(x=spot, line_dash="dash", line_color="cyan", line_width=2,
                   annotation_text=f"Spot: ${spot:.2f}")
    
    fig3.update_layout(
        height=500,
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='white'),
        xaxis=dict(title="Strike", gridcolor='#1f1f1f', tickformat='$.0f'),
        yaxis=dict(title="Gamma Exposure", gridcolor='#1f1f1f'),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig3, use_container_width=True)

with tab2:
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
st.caption(f"💾 Data: Yahoo Finance | ⏰ {datetime.now().strftime('%H:%M:%S %Y-%m-%d')} | ⚠️ Educational purposes only")
if fallback_iv:
    st.caption(f"📊 Using estimated IV: {fallback_iv:.2%}")
