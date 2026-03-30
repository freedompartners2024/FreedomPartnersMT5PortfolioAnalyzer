import streamlit as st
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")

# =========================
# PARSER MT5 ROBUSTO
# =========================
def parse_mt5_html(file):
    soup = BeautifulSoup(file, "html.parser")
    tables = soup.find_all("table")

    df = pd.read_html(str(tables))[0]

    # FIX MULTIINDEX (alcuni MT5 report)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    df = df.dropna()

    # rilevamento automatico colonne
    profit_col = [c for c in df.columns if "profit" in str(c).lower()][0]
    time_col = [c for c in df.columns if "time" in str(c).lower()][0]

    df[profit_col] = pd.to_numeric(df[profit_col], errors="coerce")
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    df = df.rename(columns={
        profit_col: "Profit",
        time_col: "Time"
    })

    df = df.dropna(subset=["Profit"])

    return df


# =========================
# METRICHE PRINCIPALI
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
# CORRELAZIONE STRATEGIE
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

    if m["pf"] > 1.5:
        score += 20
    elif m["pf"] > 1.3:
        score += 10

    if m["sharpe"] > 1:
        score += 20
    elif m["sharpe"] > 0.5:
        score += 10

    if abs(m["dd"]) < 10:
        score += 20
    elif abs(m["dd"]) < 20:
        score += 10

    if m["stagnation"] < 50:
        score += 20
    elif m["stagnation"] < 100:
        score += 10

    if m["rr"] > 1.2:
        score += 20

    return score


# =========================
# PORTFOLIO BUILDER
# =========================
def build_portfolio(metrics_list, corr, names):

    selected = []
    scores = []

    # score singole strategie
    for i, m in enumerate(metrics_list):
        s = score_strategy(m)
        scores.append(s)

        if s >= 50:
            selected.append((names[i], m, s))

    # riduzione correlazione
    final = []
    for name, m, s in selected:
        keep = True

        for f in final:
            if corr.loc[name, f[0]] > 0.7:
                keep = False

        if keep:
            final.append((name, m, s))

    # pesi
    total_score = sum([f[2] for f in final]) if final else 1

    weights = {
        f[0]: f[2] / total_score for f in final
    }

    return final, weights


# =========================
# EQUITY PORTAFOGLIO
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
st.title("🚀 Hedge Fund Portfolio Analyzer V3")

files = st.file_uploader("Carica report MT5 HTML", accept_multiple_files=True)

if files:

    metrics_list = []
    equities = {}

    # analisi singole strategie
    for file in files:
        df = parse_mt5_html(file)
        m = calculate_metrics(df)

        metrics_list.append(m)
        equities[file.name] = m["equity"]

        st.subheader(file.name)
        st.write(m)

    # correlazione
    corr = calculate_correlation(equities)

    st.subheader("🔗 Correlazione strategie")
    st.dataframe(corr)

    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, ax=ax)
    st.pyplot(fig)

    # portfolio builder
    names = list(equities.keys())
    final, weights = build_portfolio(metrics_list, corr, names)

    st.subheader("🤖 Strategie selezionate")

    for f in final:
        st.write(f"✅ {f[0]} | Score: {f[2]}")

    st.subheader("💰 Allocazione capitale")

    for k, v in weights.items():
        st.write(f"{k}: {v*100:.2f}%")

    # equity portfolio
    peq = portfolio_equity(equities, weights)

    st.subheader("📈 Equity Portfolio")
    st.line_chart(peq)

    # SCORE FINALE PORTAFOGLIO
    total_score = np.mean([f[2] for f in final]) if final else 0

    st.subheader("🏆 Score Portafoglio")

    st.metric("Score", f"{total_score:.1f}/100")

    if total_score > 75:
        st.success("PORTAFOGLIO HEDGE FUND LEVEL 🟢")
    elif total_score > 60:
        st.warning("Portafoglio buono ma migliorabile 🟡")
    else:
        st.error("Portafoglio debole 🔴")
