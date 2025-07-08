
import streamlit as st
import pandas as pd
import plotly.express as px
import pickle

st.set_page_config(layout="wide")
st.title("Trading Strategy Dashboard")

# --- Load Data ---
@st.cache_data
def load_data():
    try:
        with open('equity_history.pkl', 'rb') as f:
            equity_history = pickle.load(f)
        with open('trades_log.pkl', 'rb') as f:
            trades_log = pickle.load(f)
        return equity_history, trades_log
    except FileNotFoundError:
        st.error("Data files (equity_history.pkl, trades_log.pkl) not found. Please run multi_symbol_ml_backtest.py first.")
        return None, None

equity_history, trades_log = load_data()

if equity_history is not None and trades_log is not None:
    # --- Performance Metrics ---
    initial_capital = equity_history[0]
    final_capital = equity_history[-1]
    total_return = (final_capital / initial_capital - 1) * 100

    # Calculate Max Drawdown
    peak = equity_history[0]
    max_drawdown = 0
    for capital in equity_history:
        if capital > peak:
            peak = capital
        drawdown = (peak - capital) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    total_trades = len([t for t in trades_log if 'open' in t['type']])

    st.header("Overall Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Initial Capital", f"${initial_capital:,.2f}")
    col2.metric("Final Capital", f"${final_capital:,.2f}")
    col3.metric("Total Return", f"{total_return:.2f}%")
    col4.metric("Max Drawdown", f"{max_drawdown:.2%}")

    # --- Equity Curve ---
    st.header("Equity Curve")
    equity_df = pd.DataFrame({'Capital': equity_history})
    fig_equity = px.line(equity_df, y='Capital', title='Portfolio Equity Over Time')
    st.plotly_chart(fig_equity, use_container_width=True)

    # --- Trade Log ---
    st.header("Trade Log")
    trades_df = pd.DataFrame(trades_log)
    st.dataframe(trades_df)

    # --- Trade Summary by Symbol ---
    st.header("Trade Summary by Symbol")
    if not trades_df.empty:
        open_trades = trades_df[trades_df['type'].str.contains('open')].groupby('symbol').size().reset_index(name='Open Trades')
        close_trades = trades_df[trades_df['type'].str.contains('close')].groupby('symbol').size().reset_index(name='Close Trades')
        
        summary_df = pd.merge(open_trades, close_trades, on='symbol', how='outer').fillna(0)
        st.dataframe(summary_df)
    else:
        st.info("No trades recorded.")

else:
    st.info("Please run the backtest script to generate data.")
