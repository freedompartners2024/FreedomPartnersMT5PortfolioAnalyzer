import streamlit as st
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")

# =========================
# PARSER ROBUSTO MT5
# =========================
def parse_mt5_html(file):
    soup = BeautifulSoup(file, "html.parser")
    tables = soup.find_all("table")

    df = pd.read_html(str(tables))[0]

    # MultiIndex fix
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    df = df.dropna(how="all")

    cols = [str(c).lower() for c in df.columns]

    # =========================
    # PROFIT COLUMN DETECTION
    # =========================
    profit_candidates = [
        "profit", "p/l", "pl", "gain", "net profit", "profit/loss", "result"
    ]

    profit_col = None
    for p in profit_candidates:
        for c in df.columns:
            if p in str(c).lower():
                profit_col = c
                break
        if profit_col is not None:
            break

    if profit_col is None:
        st.error("❌ Nessuna colonna profit trovata nel file MT5")
        st.write("Colonne disponibili:", df.columns)
        st.stop()

    # =========================
    # TIME COLUMN DETECTION
    # =========================
    time_candidates = ["time", "date", "open time", "close time"]

    time_col = None
    for t in time_candidates:
        for c in df.columns:
            if t in str(c).lower():
                time_col = c
                break
        if time_col is not None:
            break

    if time_col is None:
        st.error("❌ Nessuna colonna time trovata nel file MT5")
        st.write("Colonne disponibili:", df.columns)
        st.stop()

    df[profit_col] = pd.to_numeric(df[profit_col], errors="coerce")
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    df = df.rename(columns={
        profit_col: "Profit",
        time_col: "Time"
    })

    df = df.dropna(subset=["Profit"])

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
st.title("🚀 Hedge Fund Portfolio Analyzer V4 (Stable)")

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
