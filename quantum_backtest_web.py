"""
QUANTUM PREDICTOR — BACKTESTING MODULE
Testea accuracy del modelo por ticker, frecuencia y horizonte
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import streamlit as st

st.set_page_config(
    page_title="Quantum Predictor — Backtest",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
<style>
body, .stApp { background-color: #0a0f1e; color: #c9d1d9; }
div[data-testid="metric-container"] {
    background-color: #0d1528;
    border: 1px solid #1e2a3a;
    border-radius: 6px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center; padding:10px 0 4px 0;'>
  <span style='color:#00d4ff; font-family:Courier New; font-size:1.4rem; font-weight:bold;'>
    🔬 QUANTUM PREDICTOR — BACKTESTING
  </span><br>
  <span style='color:#4a5568; font-family:Courier New; font-size:0.75rem;'>
    Accuracy real del modelo por ticker, intervalo y horizonte
  </span>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# MOTOR v2 — igual que el predictor original
# ══════════════════════════════════════════════

def descargar_datos(ticker, periodo, intervalo):
    df = yf.download(ticker, period=periodo, interval=intervalo,
                     auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"Sin datos para {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df = df[["open","high","low","close","volume"]].dropna()
    if intervalo in ["5m","15m"]:
        df = df.between_time("09:30","15:55")
    df["bar_of_day"] = df.groupby(df.index.date).cumcount()
    return df

def calcular_features(df):
    f = pd.DataFrame(index=df.index)
    c,h,l,o,v = df["close"],df["high"],df["low"],df["open"],df["volume"]
    for p in [1,2,3,5,8,13]: f[f"ret_{p}"] = c.pct_change(p)
    for p in [5,10,20]: f[f"mom_{p}"] = c/c.shift(p)-1
    for p in [9,21,50,89]:
        ema = c.ewm(span=p,adjust=False).mean()
        f[f"dist_ema_{p}"] = (c-ema)/ema
        f[f"ema_{p}"] = ema
    tp = (h+l+c)/3
    cum_tpv = (tp*v).groupby(df.index.date).cumsum()
    cum_vol  = v.groupby(df.index.date).cumsum()
    vwap = cum_tpv/cum_vol
    f["dist_vwap"]  = (c-vwap)/vwap
    f["above_vwap"] = (c>vwap).astype(int)
    df["vwap"] = vwap
    for p in [7,14,21]:
        delta = c.diff()
        g  = delta.clip(lower=0).ewm(com=p-1,adjust=False).mean()
        lo = (-delta.clip(upper=0)).ewm(com=p-1,adjust=False).mean()
        f[f"rsi_{p}"] = 100-(100/(1+g/lo.replace(0,np.nan)))
    for fast,slow,sig in [(12,26,9),(5,13,5)]:
        ml = c.ewm(span=fast,adjust=False).mean()-c.ewm(span=slow,adjust=False).mean()
        ms = ml.ewm(span=sig,adjust=False).mean()
        f[f"macd_{fast}_{slow}"] = ml-ms
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=14,adjust=False).mean()
    f["atr_norm"] = df["atr"]/c
    for p in [10,20]: f[f"vol_ratio_{p}"] = v/v.rolling(p).mean()
    f["body"]       = (c-o)/(h-l+1e-9)
    f["is_bullish"] = (c>o).astype(int)
    f["bar_norm"]   = df["bar_of_day"]/78
    f["gap"]        = o/c.shift(1)-1
    hist_vol = c.pct_change().rolling(20).std()*np.sqrt(252)*100
    f["vix_proxy"]  = hist_vol
    c_z = (c-c.rolling(20).mean())/(c.rolling(20).std()+1e-9)
    f["zscore_20"]  = c_z
    f["mean_rev"]   = -np.sign(c_z)*(c_z.abs()>1.5).astype(int)
    f["open_mom"]   = c/c.groupby(df.index.date).transform("first")-1
    return f.replace([np.inf,-np.inf],np.nan).ffill().fillna(0), df

class Stump:
    def fit(self,X,y,w=None):
        if w is None: w=np.ones(len(y))/len(y)
        best=-np.inf; self.f,self.t,self.l,self.r=0,0,1,1
        for fi in range(X.shape[1]):
            for t in np.percentile(X[:,fi],[20,40,60,80]):
                lm=X[:,fi]<=t; rm=~lm
                if lm.sum()<3 or rm.sum()<3: continue
                pl=np.sign(np.dot(w[lm],y[lm])) or 1
                pr=np.sign(np.dot(w[rm],y[rm])) or 1
                el=np.dot(w[lm],(y[lm]!=pl).astype(float))
                er=np.dot(w[rm],(y[rm]!=pr).astype(float))
                g=w.sum()-el-er
                if g>best: best=g; self.f,self.t,self.l,self.r=fi,t,pl,pr
        return self
    def predict(self,X):
        return np.where(X[:,self.f]<=self.t,self.l,self.r)

def predecir_proba(X_train, y_train, X_pred):
    valid = ~np.isnan(y_train)
    if valid.sum() < 50:
        return 0.5
    stumps, alphas = [], []
    w = np.ones(valid.sum()) / valid.sum()
    Xv, yv = X_train[valid], y_train[valid].astype(int)
    for _ in range(60):
        s = Stump().fit(Xv, yv, w)
        p = s.predict(Xv)
        e = np.clip(np.dot(w, (p != yv).astype(float)), 1e-10, 1-1e-10)
        a = 0.5 * np.log((1-e)/e)
        w *= np.exp(-a * yv * p); w /= w.sum()
        stumps.append(s); alphas.append(a)
    score = sum(float(a)*float(s.predict(X_pred).flat[0])
                for a, s in zip(alphas, stumps))
    return float(1 / (1 + np.exp(-2*score)))

# ══════════════════════════════════════════════
# WALK-FORWARD BACKTEST
# ══════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def run_backtest(ticker, periodo, intervalo, velas, prob_minima=0.70,
                 train_size=0.70, min_train=100):
    """
    Walk-forward backtest:
    - Entrena con el 70% inicial
    - En cada paso predice y compara con el resultado real
    - Avanza de a 1 barra
    """
    df_raw = descargar_datos(ticker, periodo, intervalo)
    features, df = calcular_features(df_raw)

    X = features.values
    fut = df["close"].pct_change(velas).shift(-velas)
    y_true_dir = np.sign(fut.values.flatten())
    y_true_dir[y_true_dir == 0] = 1

    n = len(X)
    start = max(min_train, int(n * train_size))

    resultados = []
    for i in range(start, n - velas):
        X_train = X[:i]
        y_train = y_true_dir[:i]
        X_pred  = X[i:i+1]

        prob = predecir_proba(X_train, y_train, X_pred)

        if prob >= prob_minima:
            senal = 1       # LONG
        elif prob <= 1 - prob_minima:
            senal = -1      # SHORT
        else:
            senal = 0       # NEUTRAL

        real = y_true_dir[i]  # direccion real

        resultados.append({
            "fecha":    df.index[i],
            "prob":     prob,
            "senal":    senal,
            "real":     real,
            "precio":   float(df["close"].iloc[i]),
            "ret_real": float(fut.iloc[i]) if not pd.isna(fut.iloc[i]) else np.nan,
        })

    if not resultados:
        return pd.DataFrame(), {}

    res = pd.DataFrame(resultados).dropna(subset=["ret_real"])

    # Solo evaluar señales no neutrales
    activas = res[res["senal"] != 0].copy()
    if activas.empty:
        return res, {}

    activas["correcto"] = (activas["senal"] == activas["real"]).astype(int)
    activas["ret_operacion"] = activas["senal"] * activas["ret_real"]

    # Metricas
    total      = len(activas)
    correctas  = activas["correcto"].sum()
    accuracy   = correctas / total if total > 0 else 0
    longs      = activas[activas["senal"] ==  1]
    shorts     = activas[activas["senal"] == -1]
    ret_total  = activas["ret_operacion"].sum() * 100
    ret_medio  = activas["ret_operacion"].mean() * 100
    win_rate_l = longs["correcto"].mean()  if len(longs)  > 0 else np.nan
    win_rate_s = shorts["correcto"].mean() if len(shorts) > 0 else np.nan

    # Sharpe simplificado
    rets = activas["ret_operacion"].dropna()
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else np.nan

    metricas = {
        "total_señales": total,
        "accuracy":      accuracy,
        "ret_total_pct": ret_total,
        "ret_medio_pct": ret_medio,
        "win_rate_long":  win_rate_l,
        "win_rate_short": win_rate_s,
        "n_longs":        len(longs),
        "n_shorts":       len(shorts),
        "sharpe":         sharpe,
        "neutras":        len(res[res["senal"] == 0]),
    }
    return res, metricas

# ══════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ Configuracion")

    tickers_input = st.text_input(
        "Tickers (separados por coma)",
        value="SPY,QQQ,TQQQ,NVDA,AAPL"
    )

    st.markdown("**Combinaciones a testear:**")
    intervalos = st.multiselect(
        "Intervalos",
        ["5m","1d"],
        default=["1d"]
    )
    horizontes = st.multiselect(
        "Horizontes (velas)",
        [1, 3, 5, 10, 21],
        default=[1, 3, 5]
    )
    prob_minima = st.slider("Probabilidad minima señal", 0.55, 0.85, 0.70, 0.05)
    run_btn = st.button("▶ Correr Backtest", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style='color:#4a5568; font-size:0.72rem; font-family:Courier New;'>
    <b style='color:#8892a4;'>Como leer los resultados:</b><br><br>
    <b>Accuracy</b>: % de señales correctas<br>
    <b>Ret. Total</b>: retorno acumulado si seguiste todas las señales<br>
    <b>Win Rate L/S</b>: accuracy separado por LONG y SHORT<br>
    <b>Sharpe</b>: retorno ajustado por riesgo<br>
    <b>Señales</b>: cuantas veces el modelo supero el umbral<br><br>
    <b style='color:#00ff88;'>Accuracy > 55%</b> = util<br>
    <b style='color:#00ff88;'>Accuracy > 60%</b> = muy bueno<br>
    <b style='color:#ffcc00;'>Accuracy 50-55%</b> = marginal<br>
    <b style='color:#ff3366;'>Accuracy < 50%</b> = no usar
    </div>
    """, unsafe_allow_html=True)

if not run_btn:
    st.info("Configura los parametros en el panel izquierdo y presiona **▶ Correr Backtest**.")
    st.stop()

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
if not tickers or not intervalos or not horizontes:
    st.warning("Ingresa al menos un ticker, intervalo y horizonte.")
    st.stop()

# Configuracion de periodos por intervalo
PERIODOS = {"5m": "60d", "1d": "2y"}

# Correr todos los tests
total_tests = len(tickers) * len(intervalos) * len(horizontes)
progress = st.progress(0, text="Corriendo backtests...")
resultados_tabla = []
i_test = 0

for ticker in tickers:
    for intervalo in intervalos:
        for horizonte in horizontes:
            i_test += 1
            progress.progress(
                i_test / total_tests,
                text=f"Testeando {ticker} | {intervalo} | {horizonte} velas..."
            )
            try:
                _, metricas = run_backtest(
                    ticker, PERIODOS[intervalo], intervalo,
                    horizonte, prob_minima
                )
                if metricas:
                    acc = metricas["accuracy"]
                    if acc >= 0.60:       rating = "🟢 Excelente"
                    elif acc >= 0.55:     rating = "🟡 Util"
                    elif acc >= 0.50:     rating = "🟠 Marginal"
                    else:                 rating = "🔴 No usar"

                    resultados_tabla.append({
                        "Ticker":       ticker,
                        "Intervalo":    intervalo,
                        "Horizonte":    f"{horizonte} velas",
                        "Rating":       rating,
                        "Accuracy":     f"{acc*100:.1f}%",
                        "Ret. Total":   f"{metricas['ret_total_pct']:+.1f}%",
                        "Ret. Medio":   f"{metricas['ret_medio_pct']:+.3f}%",
                        "Win L":        f"{metricas['win_rate_long']*100:.1f}%" if not pd.isna(metricas['win_rate_long']) else "N/A",
                        "Win S":        f"{metricas['win_rate_short']*100:.1f}%" if not pd.isna(metricas['win_rate_short']) else "N/A",
                        "Sharpe":       f"{metricas['sharpe']:.2f}" if not pd.isna(metricas['sharpe']) else "N/A",
                        "N Señales":    metricas['total_señales'],
                        "N Longs":      metricas['n_longs'],
                        "N Shorts":     metricas['n_shorts'],
                        "N Neutras":    metricas['neutras'],
                    })
            except Exception as e:
                resultados_tabla.append({
                    "Ticker": ticker, "Intervalo": intervalo,
                    "Horizonte": f"{horizonte} velas",
                    "Rating": "❌ Error", "Accuracy": str(e)[:40],
                    "Ret. Total":"—","Ret. Medio":"—",
                    "Win L":"—","Win S":"—","Sharpe":"—",
                    "N Señales":0,"N Longs":0,"N Shorts":0,"N Neutras":0,
                })

progress.empty()

if not resultados_tabla:
    st.error("No se pudieron generar resultados.")
    st.stop()

df_res = pd.DataFrame(resultados_tabla)

# ── RESUMEN GLOBAL ──
st.markdown("## 📊 Resultados del Backtest")
st.dataframe(df_res, use_container_width=True, height=400)

# ── MEJORES COMBINACIONES ──
st.markdown("## 🏆 Mejores combinaciones")
df_num = df_res[df_res["Accuracy"].str.contains("%")].copy()
df_num["acc_num"] = df_num["Accuracy"].str.replace("%","").astype(float)
df_num["ret_num"] = df_num["Ret. Total"].str.replace("%","").str.replace("+","").astype(float)
top = df_num.sort_values("acc_num", ascending=False).head(5)

for _, row in top.iterrows():
    acc_num = float(row["acc_num"])
    color = "#00ff88" if acc_num >= 60 else ("#ffcc00" if acc_num >= 55 else "#ff3366")
    st.markdown(f"""
    <div style='background:#0d1528; border-radius:8px; padding:12px; margin-bottom:8px;
                border-left:4px solid {color};'>
      <span style='color:{color}; font-family:Courier New; font-size:1rem; font-weight:bold;'>
        {row['Ticker']} | {row['Intervalo']} | {row['Horizonte']}
      </span>
      <span style='color:#8892a4; font-family:Courier New; font-size:0.8rem; margin-left:16px;'>
        {row['Rating']}
      </span><br>
      <span style='color:#c9d1d9; font-family:Courier New; font-size:0.85rem;'>
        Accuracy: <b style='color:{color};'>{row['Accuracy']}</b> &nbsp;|&nbsp;
        Ret. Total: <b>{row['Ret. Total']}</b> &nbsp;|&nbsp;
        Sharpe: <b>{row['Sharpe']}</b> &nbsp;|&nbsp;
        Señales: <b>{row['N Señales']}</b>
      </span>
    </div>
    """, unsafe_allow_html=True)

# ── COMPARACION POR TICKER ──
st.markdown("## 📈 Comparacion por Ticker")
df_pivot = df_num.pivot_table(
    index="Ticker",
    columns="Horizonte",
    values="acc_num",
    aggfunc="mean"
)
if not df_pivot.empty:
    st.dataframe(df_pivot.style.format("{:.1f}%").background_gradient(
        cmap="RdYlGn", vmin=45, vmax=65
    ), use_container_width=True)

# ── DOWNLOAD ──
csv = df_res.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇ Descargar resultados CSV",
    data=csv,
    file_name=f"backtest_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv"
)

st.caption(f"Backtest completado: {datetime.now().strftime('%H:%M:%S')} | "
           f"Prob. minima: {prob_minima:.0%} | "
           f"Walk-forward sobre datos historicos reales")
