import streamlit as st
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
import re

st.set_page_config(layout="wide")


# =========================
# PARSER UNIVERSALE MT5
# =========================
def parse_mt5_html(file):
    content = file.read().decode("utf-8", errors="ignore")
    soup = BeautifulSoup(content, "html.parser")

    text = soup.get_text(" ")

    # =========================
    # ESTRAZIONE NUMERI (TRADE PROFITS)
    # =========================
    numbers = re.findall(r"-?\d+\.\d+|-?\d+", text)

    numbers = [float(n) for n in numbers]

    if len(numbers) < 10:
        st.error("❌ Report MT5 non contiene abbastanza dati numerici")
        st.stop()

    # =========================
    # HEURISTIC: PROBABLE PROFIT SERIES
    # =========================
    # MT5 reports often contain:
    # balance/equity/profit mixed → we isolate volatility-like series

    series = pd.Series(numbers)

    # rimuove estremi assurdi (equity balance etc.)
    q_low = series.quantile(0.05)
    q_high = series.quantile(0.95)

    filtered = series[(series > q_low) & (series < q_high)]

    # differenza → simuliamo trades PnL
    pnl = filtered.diff().dropna()

    # cleanup
    pnl = pnl[(pnl < pnl.quantile(0.99)) & (pnl > pnl.quantile(0.01))]

    if len(pnl) < 5:
        st.error("❌ Impossibile ricostruire trades dal report MT5")
        st.stop()

    df = pd.DataFrame({
        "Profit": pnl.values
    })

    return df


# =========================
# METRICS
# =========================
def calculate_metrics(df):
    profit = df["Profit"]

    gross_profit = profit[profit > 0].sum()
    gross_loss = abs(profit[profit < 0].sum())

    pf = gross_profit / gross_loss if gross_loss != 0 else 0
    winrate = (profit > 0).mean() * 100

    rr = (
        profit[profit > 0].mean() / abs(profit[profit < 0].mean())
        if len(profit[profit < 0]) > 0 and len(profit[profit > 0]) > 0
        else 0
    )

    equity = profit.cumsum()
    peak = equity.cummax()
    dd = (equity - peak).min()

    sharpe = profit.mean() / profit.std() if profit.std() != 0 else 0

    stagnation = (~(equity.cummax() == equity)).sum()

    return {
        "pf": pf,
        "winrate": winrate,
        "rr": rr,
        "dd": dd,
        "sharpe": sharpe,
        "stagnation": stagnation,
        "equity": equity
    }


# =========================
# CORRELATION
# =========================
def calculate_correlation(equities):
    df = pd.DataFrame(equities).fillna(0)
    returns = df.diff().fillna(0)
    return returns.corr()


# =========================
# SCORE HEDGE FUND
# =========================
def score_strategy(m):
    score = 0

    if m["pf"] > 1.5: score += 20
    elif m["pf"] > 1.3: score += 10

    if m["sharpe"] > 1: score += 20
    elif m["sharpe"] > 0.5: score += 10

    if abs(m["dd"]) < 10: score += 20
    elif abs(m["dd"]) < 20: score += 10

    if m["stagnation"] < 50: score += 20
    elif m["stagnation"] < 100: score += 10

    if m["rr"] > 1.2: score += 20

    return score


# =========================
# PORTFOLIO BUILDER
# =========================
def build_portfolio(metrics_list, corr, names):

    selected = []

    for i, m in enumerate(metrics_list):
        s = score_strategy(m)
        if s >= 50:
            selected.append((names[i], m, s))

    final = []

    for name, m, s in selected:
        keep = True

        for f in final:
            if name in corr.columns and f[0] in corr.columns:
                if corr.loc[name, f[0]] > 0.7:
                    keep = False

        if keep:
            final.append((name, m, s))

    total_score = sum([f[2] for f in final]) if final else 1
    weights = {f[0]: f[2] / total_score for f in final}

    return final, weights


# =========================
# PORTFOLIO EQUITY
# =========================
def portfolio_equity(equities, weights):
    df = pd.DataFrame(equities).fillna(0)
    returns = df.diff().fillna(0)

    w = np.array([weights.get(k, 0) for k in df.columns])
    port = returns.dot(w)

    return port.cumsum()


# =========================
# UI
# =========================
st.title("🚀 MT5 Hedge Fund Analyzer V6 (Universal Parser)")

files = st.file_uploader("Carica report MT5 HTML", accept_multiple_files=True)

if files:

    metrics_list = []
    equities = {}

    for file in files:
        df = parse_mt5_html(file)
        m = calculate_metrics(df)

        metrics_list.append(m)
        equities[file.name] = m["equity"]

        st.subheader(file.name)
        st.write(m)

    corr = calculate_correlation(equities)

    st.subheader("🔗 Correlazione")
    st.dataframe(corr)

    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, ax=ax)
    st.pyplot(fig)

    names = list(equities.keys())
    final, weights = build_portfolio(metrics_list, corr, names)

    st.subheader("🤖 Strategie selezionate")
    for f in final:
        st.write(f"✅ {f[0]} | Score: {f[2]}")

    st.subheader("💰 Allocazione capitale")
    for k, v in weights.items():
        st.write(f"{k}: {v*100:.2f}%")

    peq = portfolio_equity(equities, weights)

    st.subheader("📈 Equity Portfolio")
    st.line_chart(peq)

    total_score = np.mean([f[2] for f in final]) if final else 0

    st.subheader("🏆 Score Portafoglio")
    st.metric("Score", f"{total_score:.1f}/100")

    if total_score > 75:
        st.success("PORTAFOGLIO HEDGE FUND LEVEL 🟢")
    elif total_score > 60:
        st.warning("Portafoglio buono 🟡")
    else:
        st.error("Portafoglio debole 🔴")
