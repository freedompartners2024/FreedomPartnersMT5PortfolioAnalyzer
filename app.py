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
    df = pd.DataFrame(equities)
    returns = df.diff().fillna(0)
    return returns.corr()

# =========================
# SCORING (HEDGE FUND STYLE)
# =========================
def score_strategy(m):
    score = 0

    if m['pf'] > 1.5: score += 20
    elif m['pf'] > 1.3: score += 10

    if m['sharpe'] > 1: score += 20
    elif m['sharpe'] > 0.5: score += 10

    if abs(m['dd']) < 10: score += 20
    elif abs(m['dd']) < 20: score += 10

    if m['stagnation'] < 50: score += 20
    elif m['stagnation'] < 100: score += 10

    if m['rr'] > 1.2: score += 20

    return score

# =========================
# AUTO BUILDER (VERO)
# =========================
def build_portfolio(metrics_list, corr, names):

    selected = []
    scores = []

    # STEP 1: score singole strategie
    for i, m in enumerate(metrics_list):
        s = score_strategy(m)
        scores.append(s)

        if s >= 50:
            selected.append((names[i], m, s))

    # STEP 2: riduzione correlazione
    final = []
    for i, (name, m, s) in enumerate(selected):
        keep = True
        for f in final:
            if corr.loc[name, f[0]] > 0.7:
                keep = False
        if keep:
            final.append((name, m, s))

    # STEP 3: pesi proporzionali al punteggio
    total_score = sum([f[2] for f in final])

    weights = {}
    for f in final:
        weights[f[0]] = f[2] / total_score

    return final, weights

# =========================
# PORTFOLIO EQUITY
# =========================
def portfolio_equity(equities, weights):
    df = pd.DataFrame(equities).fillna(0)
    returns = df.diff().fillna(0)

    w = np.array([weights[k] for k in df.columns])
    port = returns.dot(w)

    return port.cumsum()

# =========================
# UI
# =========================
st.title("🚀 Hedge Fund Portfolio Builder")

files = st.file_uploader("Carica report MT5 HTML", accept_multiple_files=True)

if files:
    metrics_list = []
    equities = {}

    for file in files:
        df = parse_mt5_html(file)
        m = calculate_metrics(df)

        metrics_list.append(m)
        equities[file.name] = m['equity']

        st.subheader(file.name)
        st.write(m)

    corr = calculate_correlation(equities)

    st.subheader("🔗 Correlazione")
    st.dataframe(corr)

    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, ax=ax)
    st.pyplot(fig)

    st.subheader("🤖 Costruzione Portfolio Automatica")

    names = list(equities.keys())
    final, weights = build_portfolio(metrics_list, corr, names)

    st.write("Strategie selezionate:")

    for f in final:
        st.write(f"✅ {f[0]} | Score: {f[2]}")

    st.write("Allocazione capitale:")

    for k, v in weights.items():
        st.write(f"{k}: {v*100:.2f}%")

    peq = portfolio_equity(equities, weights)

    st.subheader("📈 Equity Finale")
    st.line_chart(peq)

    # SCORE FINALE
    total_score = np.mean([f[2] for f in final])

    st.subheader("🏆 Score Portafoglio")
    st.metric("Score", f"{total_score:.1f}/100")

    if total_score > 75:
        st.success("Portafoglio forte (livello hedge fund)")
    elif total_score > 60:
        st.warning("Portafoglio buono ma migliorabile")
    else:
        st.error("Portafoglio debole, da rivedere")
