import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import io
warnings.filterwarnings('ignore')

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="FCF Screener & DCF Valuator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CACHED DATA FETCHING ----------
import requests

@st.cache_data(ttl=3600, show_spinner=False)
def get_sp400_tickers():
    """
    Scrape S&P 400 MidCap tickers from Wikipedia with fallback.
    Returns list of tickers, or empty list if fetch fails.
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Attempt to parse HTML tables
        tables = pd.read_html(response.text)
        if tables and len(tables) > 0 and 'Symbol' in tables[0].columns:
            tickers = tables[0]['Symbol'].tolist()
            if tickers:
                return tickers
    except requests.exceptions.Timeout:
        st.warning("⏱️ Wikipedia request timed out. Using fallback ticker list.")
    except requests.exceptions.ConnectionError:
        st.warning("🔌 Connection error fetching Wikipedia. Using fallback ticker list.")
    except Exception as e:
        st.warning(f"⚠️ Could not fetch S&P 400 from Wikipedia ({type(e).__name__}). Upload a CSV or use other sources.")
    
    # Fallback: Return empty list if Wikipedia fails
    return []

def validate_and_clean_tickers(ticker_list):
    """
    Clean and validate ticker list.
    Removes None, empty strings, and non-string values.
    Returns list of valid uppercase tickers.
    """
    if not isinstance(ticker_list, list):
        return []
    
    valid_tickers = []
    for ticker in ticker_list:
        # Skip None values
        if ticker is None:
            continue
        # Convert to string and strip whitespace
        ticker_str = str(ticker).strip().upper()
        # Skip empty strings
        if ticker_str:
            valid_tickers.append(ticker_str)
    
    return valid_tickers

# Small-cap sample: you can replace this with a CSV read or an ETF holdings scrape
SMALL_CAP_TICKERS = ['SITC', 'CRNC', 'AMSC', 'OSUR', 'AXNX', 'SMCI', 'CROX', 'MOD', 'QLYS']

@st.cache_data(ttl=3600, show_spinner=False)
def get_fcf_data(ticker):
    """
    Fetch the last 3 years' annual Free Cash Flow (CFO - CapEx).
    Returns a Series with dates as index, or None if not available.
    """
    stock = yf.Ticker(ticker)
    try:
        cf = stock.cashflow
        if cf.empty:
            return None
        # Locate rows
        cfo_row = cf[cf.index.str.lower().str.contains('operating cash flow|cash from operations|total cash from operating')].iloc[0]
        capex_row = cf[cf.index.str.lower().str.contains('capital expenditures|purchase of property')].iloc[0]
        # capex is normally negative, so FCF = cfo + capex (if negative)
        fcf_series = cfo_row + capex_row  # works because capex is negative
        fcf_series = fcf_series.dropna()
        if len(fcf_series) < 3:
            return None
        # Return last 3 sorted oldest to newest
        fcf_series = fcf_series.sort_index(ascending=True).iloc[-3:]
        return fcf_series
    except:
        return None

@st.cache_data(ttl=1800, show_spinner=False)
def get_enterprise_value(ticker):
    """Return Enterprise Value = Market Cap + Total Debt - Cash."""
    stock = yf.Ticker(ticker)
    try:
        info = stock.info
        market_cap = info.get('marketCap')
        total_debt = info.get('totalDebt')
        cash = info.get('totalCash')
        if None in (market_cap, total_debt, cash):
            return None
        return market_cap + total_debt - cash
    except:
        return None

@st.cache_data(ttl=1800, show_spinner=False)
def get_market_cap(ticker):
    stock = yf.Ticker(ticker)
    try:
        return stock.info.get('marketCap')
    except:
        return None

@st.cache_data(ttl=1800, show_spinner=False)
def get_current_price(ticker):
    stock = yf.Ticker(ticker)
    try:
        return stock.info.get('currentPrice') or stock.info.get('regularMarketPreviousClose')
    except:
        return None

@st.cache_data(ttl=1800, show_spinner=False)
def get_company_name(ticker):
    stock = yf.Ticker(ticker)
    try:
        return stock.info.get('shortName') or stock.info.get('longName') or ticker
    except:
        return ticker

@st.cache_data(ttl=3600, show_spinner=False)
def get_shares_outstanding(ticker):
    stock = yf.Ticker(ticker)
    try:
        return stock.info.get('sharesOutstanding')
    except:
        return None

def fcf_yield(ticker):
    """Returns the average FCF / EV yield as a percentage."""
    fcf = get_fcf_data(ticker)
    ev = get_enterprise_value(ticker)
    if fcf is None or ev is None or ev <= 0:
        return None
    avg_fcf = fcf.mean()
    if avg_fcf <= 0:
        return None
    return (avg_fcf / ev) * 100

# ---------- DCF FUNCTIONS ----------
def dcf_valuation(ticker, growth_stage1, years_stage1, terminal_growth, wacc, manual_fcf=None, manual_shares=None):
    """
    Two-stage DCF. Returns a dict of results.
    """
    # Fetch trailing FCF (most recent annual)
    fcf_data = get_fcf_data(ticker)
    if fcf_data is None or len(fcf_data) == 0:
        return None
    fcf0 = fcf_data.iloc[-1]  # most recent

    shares = get_shares_outstanding(ticker)
    if shares is None or shares <= 0:
        return None

    if manual_fcf is not None:
        fcf0 = manual_fcf
    if manual_shares is not None:
        shares = manual_shares

    # Project stage 1
    fcf = fcf0
    fcf_stage1 = []
    discount_factors = []
    for i in range(1, years_stage1 + 1):
        fcf *= (1 + growth_stage1)
        fcf_stage1.append(fcf)
        discount_factors.append(1 / (1 + wacc) ** i)

    pv_stage1 = sum(f * d for f, d in zip(fcf_stage1, discount_factors))

    # Terminal value
    final_fcf = fcf_stage1[-1]
    terminal_value = final_fcf * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal = terminal_value * discount_factors[-1]

    # Enterprise value
    enterprise_value = pv_stage1 + pv_terminal

    # Adjust for net debt
    try:
        info = yf.Ticker(ticker).info
        cash = info.get('totalCash', 0) or 0
        debt = info.get('totalDebt', 0) or 0
        net_debt = debt - cash
    except:
        net_debt = 0
    equity_value = enterprise_value - net_debt

    intrinsic_per_share = equity_value / shares
    current_price = get_current_price(ticker)
    if current_price:
        margin_safety = (1 - current_price / intrinsic_per_share) * 100
    else:
        margin_safety = None

    return {
        'Ticker': ticker,
        'Current Price': current_price,
        'Intrinsic Value/Share': intrinsic_per_share,
        'Margin of Safety (%)': margin_safety,
        'Trailing FCF': fcf0,
        'FCF/EV Yield (%)': (fcf0 / enterprise_value) * 100,
        'Stage1 PV': pv_stage1,
        'Terminal PV': pv_terminal,
        'Enterprise Value': enterprise_value,
        'Equity Value': equity_value,
        'Shares Outstanding': shares
    }

# ---------- TAB STRUCTURE ----------
tab1, tab2, tab3 = st.tabs(["📊 Screener", "🧮 DCF Valuator", "💰 FCF Analysis"])

# ========== TAB 1: SCREENER ==========
with tab1:
    st.title("🔍 FCF/EV Yield Screener")
    st.markdown("Screen small/mid-cap stocks to discover high free cash flow yield opportunities.")

    # Sidebar for screener parameters
    with st.sidebar:
        st.header("Screener Settings")
        market_cap_min = st.number_input("Min Market Cap ($M)", value=300, step=50) * 1e6
        market_cap_max = st.number_input("Max Market Cap ($M)", value=10000, step=1000) * 1e6
        min_fcf_yield = st.slider("Minimum FCF/EV Yield (%)", 0.0, 15.0, 5.0, 0.5)

        st.divider()
        st.subheader("📋 Ticker Sources")

        # Ticker source options
        use_midcap = st.checkbox("Include S&P 400 MidCaps", value=True)
        use_smallcap = st.checkbox("Include Small-Cap Sample", value=True)
        
        # File upload option
        uploaded_file = st.file_uploader(
            "Or upload CSV with tickers",
            type=['csv'],
            help="CSV should have a 'Symbol' or 'Ticker' column"
        )
        
        uploaded_tickers = []
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                # Try to find ticker column (case-insensitive)
                ticker_col = None
                for col in df_upload.columns:
                    if col.lower() in ['symbol', 'ticker']:
                        ticker_col = col
                        break
                
                if ticker_col:
                    raw_tickers = df_upload[ticker_col].tolist()
                    uploaded_tickers = validate_and_clean_tickers(raw_tickers)
                    st.success(f"✅ Loaded {len(uploaded_tickers)} valid tickers from file")
                    if len(raw_tickers) != len(uploaded_tickers):
                        st.info(f"ℹ️ Filtered out {len(raw_tickers) - len(uploaded_tickers)} invalid entries")
                else:
                    st.error(f"❌ CSV must have 'Symbol' or 'Ticker' column. Found: {list(df_upload.columns)}")
            except Exception as e:
                st.error(f"❌ Error reading CSV: {e}")
        
        # Custom tickers text input
        custom_tickers = st.text_area("Or enter custom tickers (comma-separated)", value="")

        st.divider()
        
        if st.button("Run Screener", type="primary", key="run_screener_btn"):
            st.session_state.run_screen = True

    # Main area
    if st.session_state.get('run_screen', False):
        with st.spinner("Fetching ticker list and financial data... (may take a few minutes)"):
            tickers = set()
            
            if use_midcap:
                sp400_tickers = get_sp400_tickers()
                if sp400_tickers:
                    clean_sp400 = validate_and_clean_tickers(sp400_tickers)
                    tickers.update(clean_sp400)
                    st.info(f"✅ Loaded {len(clean_sp400)} S&P 400 tickers")
                else:
                    st.warning("⚠️ S&P 400 tickers not available. Use other sources or upload a CSV.")
            
            if use_smallcap:
                clean_small = validate_and_clean_tickers(SMALL_CAP_TICKERS)
                tickers.update(clean_small)
                st.info(f"✅ Loaded {len(clean_small)} small-cap tickers")
            
            if uploaded_tickers:
                tickers.update(uploaded_tickers)
            
            if custom_tickers:
                custom_list = [t.strip().upper() for t in custom_tickers.split(',') if t.strip()]
                clean_custom = validate_and_clean_tickers(custom_list)
                tickers.update(clean_custom)
                st.info(f"✅ Loaded {len(clean_custom)} custom tickers")

            tickers = list(tickers)
            
            if not tickers:
                st.error("❌ No valid tickers selected. Please select at least one source.")
            else:
                st.write(f"Screening {len(tickers)} tickers...")

                results = []
                progress = st.progress(0)
                failed_tickers = []
                
                for i, ticker in enumerate(tickers):
                    progress.progress((i + 1) / len(tickers))
                    
                    # Market cap filter
                    try:
                        mkt_cap = get_market_cap(ticker)
                        if mkt_cap is None or mkt_cap < market_cap_min or mkt_cap > market_cap_max:
                            continue
                    except Exception as e:
                        failed_tickers.append((ticker, str(e)))
                        continue
                    
                    # FCF yield
                    try:
                        fy = fcf_yield(ticker)
                        if fy is None:
                            continue
                        # Store all results, not just filtered ones
                    except Exception as e:
                        failed_tickers.append((ticker, str(e)))
                        continue
                    
                    try:
                        company_name = get_company_name(ticker)
                        current_price = get_current_price(ticker)
                        results.append({
                            'Ticker': ticker,
                            'Company': company_name,
                            'Current Price': current_price,
                            'Market Cap ($M)': mkt_cap / 1e6,
                            'FCF/EV Yield (%)': round(fy, 2)
                        })
                    except Exception as e:
                        failed_tickers.append((ticker, str(e)))
                        continue

                if results:
                    df = pd.DataFrame(results).sort_values('FCF/EV Yield (%)', ascending=False)
                    st.session_state.screen_results = df
                    
                    # NEW: Show ALL data first, then allow filtering
                    st.success(f"✅ Fetched {len(df)} tickers with valid FCF data.")
                    
                    # Interactive filtering section
                    st.subheader("🔎 Filter Results")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        filter_min_yield = st.number_input(
                            "Filter by Min FCF/EV Yield (%)",
                            value=min_fcf_yield,
                            step=0.5,
                            key="filter_yield"
                        )
                    
                    with col2:
                        filter_min_cap = st.number_input(
                            "Filter by Min Market Cap ($M)",
                            value=market_cap_min / 1e6,
                            step=100,
                            key="filter_cap_min"
                        ) * 1e6
                    
                    with col3:
                        filter_max_cap = st.number_input(
                            "Filter by Max Market Cap ($M)",
                            value=market_cap_max / 1e6,
                            step=100,
                            key="filter_cap_max"
                        ) * 1e6
                    
                    # Apply filters
                    filtered_df = df[
                        (df['FCF/EV Yield (%)'] >= filter_min_yield) &
                        (df['Market Cap ($M)'] >= filter_min_cap / 1e6) &
                        (df['Market Cap ($M)'] <= filter_max_cap / 1e6)
                    ].sort_values('FCF/EV Yield (%)', ascending=False)
                    
                    st.subheader("📊 Results")
                    st.write(f"Showing {len(filtered_df)} of {len(df)} tickers")
                    st.dataframe(filtered_df, use_container_width=True)
                    
                    # Allow user to select a ticker for DCF deep-dive
                    if not filtered_df.empty:
                        selected = st.selectbox(
                            "Select a ticker for DCF analysis:",
                            filtered_df['Ticker'].tolist(),
                            key="screener_ticker_select"
                        )
                        if selected:
                            st.session_state.dcf_ticker = selected
                            st.info(f"✅ Selected **{selected}** for DCF analysis. Go to the 'DCF Valuator' tab.")
                
                else:
                    st.warning("❌ No stocks matched the criteria.")
                    st.session_state.screen_results = pd.DataFrame()
                
                # Show summary of failed tickers
                if failed_tickers:
                    with st.expander(f"ℹ️ {len(failed_tickers)} tickers failed to fetch (click to see)"):
                        for ticker, error in failed_tickers[:10]:  # Show first 10
                            st.caption(f"🔴 {ticker}: {error[:80]}")
                        if len(failed_tickers) > 10:
                            st.caption(f"... and {len(failed_tickers) - 10} more")

# ========== TAB 2: DCF VALUATOR ==========
with tab2:
    st.title("🧮 DCF Valuation")
    st.markdown("Run a two-stage DCF model on any ticker. Enter a ticker from the screener or any valid symbol.")

    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        dcf_ticker = st.text_input(
            "Enter ticker symbol:",
            value=st.session_state.get('dcf_ticker', ''),
            placeholder="e.g., AAPL, MSFT, or any ticker",
            key="dcf_ticker_input"
        ).upper()
    
    with col2:
        if st.button("Quick Fetch", help="Fetch data and preview"):
            if dcf_ticker:
                st.session_state.dcf_fetch_ticker = dcf_ticker
    
    if dcf_ticker and st.session_state.get('dcf_fetch_ticker') == dcf_ticker:
        # Show company info
        try:
            company_name = get_company_name(dcf_ticker)
            current_price = get_current_price(dcf_ticker)
            market_cap = get_market_cap(dcf_ticker)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Company", company_name)
            with col2:
                st.metric("Current Price", f"${current_price:.2f}" if current_price else "N/A")
            with col3:
                st.metric("Market Cap", f"${market_cap/1e9:.2f}B" if market_cap else "N/A")
            with col4:
                fy = fcf_yield(dcf_ticker)
                st.metric("FCF/EV Yield", f"{fy:.2f}%" if fy else "N/A")
        except Exception as e:
            st.warning(f"⚠️ Could not fetch preview data: {e}")
    
    st.divider()
    st.subheader("DCF Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        growth_stage1 = st.slider("Stage 1 FCF Growth Rate (%)", 0.0, 30.0, 8.0, 0.5) / 100
        years_stage1 = st.slider("Stage 1 Years", 1, 10, 5)
    with col2:
        terminal_growth = st.slider("Terminal Growth Rate (%)", 0.0, 5.0, 3.0, 0.1) / 100
        wacc = st.slider("Discount Rate (WACC %)", 5.0, 15.0, 9.0, 0.1) / 100

    if st.button("Run DCF", type="primary", key="run_dcf_btn"):
        if not dcf_ticker:
            st.error("❌ Please enter a ticker symbol.")
        else:
            with st.spinner("Calculating intrinsic value..."):
                result = dcf_valuation(dcf_ticker, growth_stage1, years_stage1, terminal_growth, wacc)
            
            if result is None:
                st.error(f"❌ Unable to fetch financial data for **{dcf_ticker}**. This ticker may not exist or lacks FCF/earnings data.")
            else:
                st.success(f"✅ DCF valuation complete for **{result['Ticker']}**")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Intrinsic Value/Share", f"${result['Intrinsic Value/Share']:.2f}")
                    st.metric("Current Price", f"${result['Current Price']:.2f}" if result['Current Price'] else "N/A")
                with col2:
                    st.metric("Margin of Safety", f"{result['Margin of Safety (%)']:.2f}%" if result['Margin of Safety'] is not None else "N/A")
                    st.metric("FCF/EV Yield", f"{result['FCF/EV Yield (%)']:.2f}%")
                with col3:
                    st.metric("Trailing FCF", f"${result['Trailing FCF']:,.0f}")
                    st.metric("Enterprise Value", f"${result['Enterprise Value']:,.0f}")

                with st.expander("📋 See DCF Breakdown"):
                    dcf_details = {
                        "Stage 1 PV of FCFs": f"${result['Stage1 PV']:,.0f}",
                        "Terminal Value PV": f"${result['Terminal PV']:,.0f}",
                        "Enterprise Value": f"${result['Enterprise Value']:,.0f}",
                        "Equity Value": f"${result['Equity Value']:,.0f}",
                        "Shares Outstanding": f"{result['Shares Outstanding']:,.0f}"
                    }
                    st.json(dcf_details)

# ========== TAB 3: FCF ANALYSIS ==========
with tab3:
    st.title("💰 FCF Analysis")
    st.markdown("Analyze Free Cash Flow for any ticker. View historical FCF trends and calculate FCF/EV yield.")

    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fcf_ticker = st.text_input(
            "Enter ticker symbol:",
            placeholder="e.g., AAPL, MSFT, or any ticker",
            key="fcf_ticker_input"
        ).upper()
    
    with col2:
        if st.button("Analyze FCF", type="primary", key="analyze_fcf_btn"):
            if fcf_ticker:
                st.session_state.fcf_analyze_ticker = fcf_ticker
    
    if fcf_ticker and st.session_state.get('fcf_analyze_ticker') == fcf_ticker:
        try:
            company_name = get_company_name(fcf_ticker)
            current_price = get_current_price(fcf_ticker)
            market_cap = get_market_cap(fcf_ticker)
            
            st.subheader(f"{company_name} ({fcf_ticker})")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"${current_price:.2f}" if current_price else "N/A")
            with col2:
                st.metric("Market Cap", f"${market_cap/1e9:.2f}B" if market_cap else "N/A")
            with col3:
                fy = fcf_yield(fcf_ticker)
                st.metric("FCF/EV Yield", f"{fy:.2f}%" if fy else "N/A")
            
            st.divider()
            
            # Fetch FCF data
            fcf_data = get_fcf_data(fcf_ticker)
            
            if fcf_data is None or len(fcf_data) == 0:
                st.error(f"❌ No FCF data available for **{fcf_ticker}**. This ticker may not have sufficient financial data.")
            else:
                # Display FCF table
                st.subheader("📊 Historical Free Cash Flow (Last 3 Years)")
                
                fcf_df = pd.DataFrame({
                    'Year': fcf_data.index.strftime('%Y-%m-%d'),
                    'FCF ($)': fcf_data.values
                }).reset_index(drop=True)
                
                fcf_df['FCF ($)'] = fcf_df['FCF ($)'].apply(lambda x: f"${x:,.0f}")
                st.dataframe(fcf_df, use_container_width=True, hide_index=True)
                
                # Calculate statistics
                fcf_values = get_fcf_data(fcf_ticker).values
                avg_fcf = np.mean(fcf_values)
                latest_fcf = fcf_values[-1]
                
                # Calculate FCF growth
                if len(fcf_values) > 1:
                    fcf_growth = ((fcf_values[-1] - fcf_values[0]) / abs(fcf_values[0])) * 100
                else:
                    fcf_growth = None
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Latest FCF", f"${latest_fcf:,.0f}")
                with col2:
                    st.metric("Average FCF (3yr)", f"${avg_fcf:,.0f}")
                with col3:
                    if fcf_growth is not None:
                        st.metric("3-Year Growth", f"{fcf_growth:.1f}%")
                    else:
                        st.metric("3-Year Growth", "N/A")
                
                st.divider()
                
                # Valuation metrics
                st.subheader("💹 Valuation Metrics")
                
                ev = get_enterprise_value(fcf_ticker)
                if ev is not None and ev > 0:
                    ev_to_fcf = ev / avg_fcf if avg_fcf > 0 else None
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Enterprise Value", f"${ev:,.0f}")
                    with col2:
                        if ev_to_fcf:
                            st.metric("EV / Avg FCF Multiple", f"{ev_to_fcf:.2f}x")
                        else:
                            st.metric("EV / Avg FCF Multiple", "N/A")
                
                # Insights
                st.subheader("📈 Insights")
                
                if latest_fcf > avg_fcf:
                    st.success(f"✅ Latest FCF (${latest_fcf:,.0f}) is **above** the 3-year average (${avg_fcf:,.0f})")
                elif latest_fcf < avg_fcf:
                    st.warning(f"⚠️ Latest FCF (${latest_fcf:,.0f}) is **below** the 3-year average (${avg_fcf:,.0f})")
                else:
                    st.info(f"ℹ️ Latest FCF equals the 3-year average")
                
                if fcf_growth is not None:
                    if fcf_growth > 0:
                        st.success(f"✅ FCF has grown {fcf_growth:.1f}% over 3 years")
                    elif fcf_growth < 0:
                        st.error(f"❌ FCF has declined {abs(fcf_growth):.1f}% over 3 years")
                    else:
                        st.info("ℹ️ FCF is flat over 3 years")
        
        except Exception as e:
            st.error(f"❌ Error fetching data for **{fcf_ticker}**: {str(e)}")

# ---------- FOOTER ----------
st.divider()
st.caption("📌 Data provided by Yahoo Finance. For educational purposes only. Not financial advice.")
