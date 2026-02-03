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
