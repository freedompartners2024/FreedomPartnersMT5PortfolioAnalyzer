import streamlit as st
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")

# =========================
# PARSER
# =========================
def parse_mt5_html(file):
    soup = BeautifulSoup(file, "html.parser")
    tables = soup.find_all("table")
    df = pd.read_html(str(tables))[1]

    df.columns = df.columns.droplevel(0)
    df = df.dropna()

    df = df[df['Profit'].notna()]
    df['Profit'] = pd.to_numeric(df['Profit'], errors='coerce')
    df['Time'] = pd.to_datetime(df['Time'])

    return df

# =========================
# METRICS
# =========================
def calculate_metrics(df):
    profit = df['Profit']

    gross_profit = profit[profit > 0].sum()
    gross_loss = abs(profit[profit < 0].sum())

    pf = gross_profit / gross_loss if gross_loss != 0 else 0
    winrate = (profit > 0).mean() * 100
    rr = profit[profit > 0].mean() / abs(profit[profit < 0].mean())

    equity = profit.cumsum()
    peak = equity.cummax()
    dd = (equity - peak).min()

    sharpe = profit.mean() / profit.std() if profit.std() != 0 else 0

    stagnation = (~(equity.cummax() == equity)).sum()

    df['Month'] = df['Time'].dt.to_period('M')
    monthly = df.groupby('Month')['Profit'].sum()

    return {
        "Profit Factor": pf,
        "Win Rate": winrate,
        "RR": rr,
        "Max DD": dd,
        "Sharpe": sharpe,
        "Stagnation": stagnation,
        "Equity": equity,
        "Monthly": monthly
    }

# =========================
# CORRELATION
# =========================
def calculate_correlation(equities):
    df = pd.DataFrame(equities)
    returns = df.diff().fillna(0)
    return returns.corr()

# =========================
# MONTE CARLO
# =========================
def monte_carlo(profits, simulations=200):
    results = []
    for _ in range(simulations):
        shuffled = np.random.permutation(profits)
        equity = np.cumsum(shuffled)
        results.append(equity[-1])
    return results

# =========================
# OPTIMIZER
# =========================
def optimize_portfolio(equities, trials=1000):
    combined = pd.DataFrame(equities).fillna(0)
    returns = combined.diff().fillna(0)

    best_score = -np.inf
    best_weights = None

    for _ in range(trials):
        weights = np.random.random(len(combined.columns))
        weights /= weights.sum()

        portfolio_returns = returns.dot(weights)
        equity = portfolio_returns.cumsum()

        dd = (equity - equity.cummax()).min()
        profit = equity.iloc[-1]

        score = profit + dd

        if score > best_score:
            best_score = score
            best_weights = weights

    return best_weights

# =========================
# AI BUILDER
# =========================
def auto_builder(metrics_list, corr, names):
    selected = []
    weights = []

    for i, m in enumerate(metrics_list):
        if m['Profit Factor'] > 1.3 and m['Sharpe'] > 0.3:
            selected.append(names[i])

    return selected

# =========================
# UI
# =========================
st.title("🚀 MT5 Portfolio Analyzer AI")

files = st.file_uploader("Carica report MT5 HTML", accept_multiple_files=True)

if files:
    metrics_list = []
    equities = {}
    combined_equity = None

    for file in files:
        df = parse_mt5_html(file)
        metrics = calculate_metrics(df)
        metrics_list.append(metrics)

        equity = metrics['Equity']
        equities[file.name] = equity

        if combined_equity is None:
            combined_equity = equity
        else:
            combined_equity = combined_equity.add(equity, fill_value=0)

        st.subheader(file.name)
        st.write({k:v for k,v in metrics.items() if k not in ['Equity','Monthly']})
        st.line_chart(equity)
        st.bar_chart(metrics['Monthly'])

    st.subheader("📈 Portfolio")
    st.line_chart(combined_equity)

    corr = calculate_correlation(equities)

    st.subheader("🔗 Correlazione")
    st.dataframe(corr)

    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, ax=ax)
    st.pyplot(fig)

    all_profits = np.concatenate([m['Equity'].diff().fillna(0).values for m in metrics_list])
    mc = monte_carlo(all_profits)

    st.subheader("🎲 Monte Carlo")
    st.bar_chart(mc)

    st.subheader("⚖️ Ottimizzazione")
    if st.button("Ottimizza"):
        weights = optimize_portfolio(equities)
        for i, name in enumerate(equities.keys()):
            st.write(f"{name}: {weights[i]*100:.2f}%")

    st.subheader("🤖 Auto Portfolio Builder")
    names = list(equities.keys())
    selected = auto_builder(metrics_list, corr, names)

    for s in selected:
        st.write(f"✅ {s}")
