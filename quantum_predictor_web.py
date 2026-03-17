"""
QUANTUM SIGNAL PREDICTOR v7.0 — WEB
Motor v2 · Monte Carlo · Noticias · Auto-refresh
Convertido a Streamlit para acceso web
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import urllib.request
import json
import time
from datetime import datetime
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Quantum Signal Predictor v7.0",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── ESTILOS ──
st.markdown("""
<style>
body, .stApp { background-color: #0a0f1e; color: #c9d1d9; }
.signal-long  { color: #00ff88; font-size: 2.8rem; font-weight: bold; font-family: 'Courier New'; }
.signal-short { color: #ff3366; font-size: 2.8rem; font-weight: bold; font-family: 'Courier New'; }
.signal-neutral { color: #ffcc00; font-size: 2.8rem; font-weight: bold; font-family: 'Courier New'; }
.metric-label { color: #4a5568; font-size: 0.75rem; font-family: 'Courier New'; }
.metric-value { color: #ffffff; font-size: 0.95rem; font-weight: bold; font-family: 'Courier New'; }
.news-alcista { color: #00ff88; font-weight: bold; }
.news-bajista { color: #ff3366; font-weight: bold; }
.news-neutral { color: #8892a4; }
.stButton > button {
    background-color: #00d4ff; color: #0a0f1e;
    font-family: 'Courier New'; font-weight: bold;
    border: none; border-radius: 4px;
}
.stButton > button:hover { background-color: #00aacc; }
div[data-testid="metric-container"] {
    background-color: #0d1528;
    border: 1px solid #1e2a3a;
    border-radius: 6px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# CONFIGURACION
# ══════════════════════════════════════════════
ATR_SL_MULT   = 1.5
ATR_TP_MULT   = 3.0
PROB_MINIMA   = 0.70
GNEWS_API_KEY = "66e2b9d63114cb38d66d47ae8007af64"

KEYWORDS_BAJISTAS = ["tariff","tariffs","sanction","ban","restrict","shutdown","default",
    "recession","crash","collapse","war","attack","impeach","rate hike",
    "trade war","deficit","layoffs","bubble","inflation surge"]
KEYWORDS_ALCISTAS = ["deal","agreement","rate cut","stimulus","trade deal","growth","jobs",
    "boom","rally","record","deregulation","fed pause","tax cut","strong"]
TRUMP_KEYWORDS = ["trump","tariff","white house","executive order","mar-a-lago","maga"]

# ══════════════════════════════════════════════
# HORIZONTE
# ══════════════════════════════════════════════
def horizonte_config(unidad, valor):
    if unidad == "min":
        return {"intervalo":"5m","periodo":"60d","velas":max(1,round(valor/5)),"label":f"{valor}min"}
    else:
        if valor <= 30:
            return {"intervalo":"1d","periodo":"5y","velas":valor,"label":f"{valor}d"}
        else:
            return {"intervalo":"1wk","periodo":"10y","velas":max(1,round(valor/5)),"label":f"{valor}d"}

# ══════════════════════════════════════════════
# DATOS
# ══════════════════════════════════════════════
def descargar_datos(ticker, cfg):
    df = yf.download(ticker, period=cfg["periodo"], interval=cfg["intervalo"],
                     auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No se encontraron datos para '{ticker}'")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [col.lower() for col in df.columns]
    df.index.name = "datetime"
    df = df[["open","high","low","close","volume"]].dropna()
    if cfg["intervalo"] in ["5m","15m"]:
        df = df.between_time("09:30","15:55")
    df["bar_of_day"] = df.groupby(df.index.date).cumcount()
    return df

# ══════════════════════════════════════════════
# FEATURES — motor v2 exacto
# ══════════════════════════════════════════════
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

# ══════════════════════════════════════════════
# REGIMEN
# ══════════════════════════════════════════════
def detectar_regimen(df):
    c = df["close"]
    ema20 = c.ewm(span=20,adjust=False).mean()
    ema50 = c.ewm(span=50,adjust=False).mean()
    tr = pd.concat([df["high"]-df["low"],(df["high"]-c.shift()).abs(),
                    (df["low"]-c.shift()).abs()],axis=1).max(axis=1)
    atr_norm = tr.ewm(span=14,adjust=False).mean()/c
    alta_vol = atr_norm > atr_norm.rolling(100).mean()*2.0
    reg = pd.Series(2,index=df.index)
    reg[~alta_vol&(ema20>ema50*1.002)] = 0
    reg[~alta_vol&(ema20<ema50*0.998)] = 1
    reg[alta_vol] = 3
    return reg.fillna(2).astype(int)

# ══════════════════════════════════════════════
# MODELO — motor v2 exacto (AdaBoost manual)
# ══════════════════════════════════════════════
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

def entrenar_predecir(df, features, velas):
    X = features.values
    fut = df["close"].pct_change(velas).shift(-velas)
    y   = np.sign(fut.values.flatten()); y[y==0]=1
    n   = int(len(X)*0.82)
    Xt,yt = X[:n],y[:n]
    valid = ~np.isnan(yt)
    if valid.sum()<50: return 0.5
    stumps,alphas,w=[],[],np.ones(valid.sum())/valid.sum()
    Xv,yv=Xt[valid],yt[valid].astype(int)
    for _ in range(60):
        s=Stump().fit(Xv,yv,w); p=s.predict(Xv)
        e=np.clip(np.dot(w,(p!=yv).astype(float)),1e-10,1-1e-10)
        a=0.5*np.log((1-e)/e)
        w*=np.exp(-a*yv*p); w/=w.sum()
        stumps.append(s); alphas.append(a)
    Xlast=X[-1:].reshape(1,-1)
    score=sum(float(a)*float(s.predict(Xlast).flat[0]) for a,s in zip(alphas,stumps))
    return float(1/(1+np.exp(-2*score)))

# ══════════════════════════════════════════════
# MONTE CARLO
# ══════════════════════════════════════════════
def monte_carlo(df, velas, sl=None, tp=None, n_sim=10000):
    c = df["close"]
    rets = c.pct_change().dropna()
    if len(rets)<20: return None
    precio = float(c.iloc[-1].item() if hasattr(c.iloc[-1],'item') else c.iloc[-1])
    mu,sigma = float(rets.mean()),float(rets.std())
    np.random.seed(42)
    shocks = np.random.normal(mu,sigma,(n_sim,velas))
    caminos = precio*np.cumprod(1+shocks,axis=1)
    pf = caminos[:,-1]
    target=float(np.percentile(pf,50))
    t_bull=float(np.percentile(pf,75))
    t_bear=float(np.percentile(pf,25))
    prob_sub=float((pf>precio).mean())
    ret_med=float((target/precio-1)*100)
    prob_tp=None
    if sl is not None and tp is not None:
        atp=astp=0
        for cam in caminos:
            for p in cam:
                if tp>precio and p>=tp:   atp+=1;  break
                elif tp<precio and p<=tp: atp+=1;  break
                elif sl>precio and p>=sl: astp+=1; break
                elif sl<precio and p<=sl: astp+=1; break
        total=atp+astp
        prob_tp=atp/total if total>0 else None
    return {"target":target,"bull":t_bull,"bear":t_bear,
            "prob_sub":prob_sub,"ret_med":ret_med,"prob_tp":prob_tp,"n_sim":n_sim}

# ══════════════════════════════════════════════
# NOTICIAS
# ══════════════════════════════════════════════
@st.cache_data(ttl=300)
def obtener_noticias():
    try:
        queries = ["Trump tariff market","stock market SPY federal reserve"]
        todas = []
        for q in queries:
            url = ("https://gnews.io/api/v4/search"
                   "?q=" + q.replace(" ", "+") +
                   "&lang=en&max=5&sortby=publishedAt"
                   "&apikey=" + GNEWS_API_KEY)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read())
            articles = data.get("articles", [])
            for art in articles:
                art["source"] = {"name": art.get("source", {}).get("name", "")}
            todas.extend(articles)
        noticias = []
        vistos = set()
        for art in todas:
            titulo = art.get("title", "") or ""
            if titulo in vistos or titulo == "[Removed]": continue
            vistos.add(titulo)
            desc  = (art.get("description", "") or "").lower()
            texto = titulo.lower() + " " + desc
            sb = sum(1 for k in KEYWORDS_BAJISTAS if k in texto)
            sa = sum(1 for k in KEYWORDS_ALCISTAS if k in texto)
            trump = any(k in texto for k in TRUMP_KEYWORDS)
            if sb > sa:   imp,col = "BAJISTA","bajista"
            elif sa > sb: imp,col = "ALCISTA","alcista"
            else:         imp,col = "NEUTRAL","neutral"
            pub = art.get("publishedAt", "")
            try:
                dt   = datetime.strptime(pub, "%Y-%m-%dT%H:%M:%SZ")
                mins = int((datetime.utcnow()-dt).total_seconds()//60)
                t    = f"hace {mins}m" if mins < 60 else f"hace {mins//60}h"
            except:
                t = ""
            noticias.append({
                "titulo":  titulo[:80] + ("..." if len(titulo) > 80 else ""),
                "impacto": imp, "css": col, "trump": trump,
                "tiempo":  t,   "fuente": art.get("source", {}).get("name", ""),
            })
        noticias.sort(key=lambda x: (not x["trump"], x["impacto"] == "NEUTRAL"))
        return noticias[:8]
    except Exception as e:
        return [{"titulo": f"Error noticias: {e}", "impacto": "NEUTRAL",
                 "css": "neutral", "trump": False, "tiempo": "", "fuente": ""}]

# ══════════════════════════════════════════════
# ANALISIS PRINCIPAL
# ══════════════════════════════════════════════
@st.cache_data(ttl=60)
def analizar(ticker, unidad, valor):
    cfg = horizonte_config(unidad, valor)
    df  = descargar_datos(ticker, cfg)
    features, df = calcular_features(df)
    reg  = detectar_regimen(df)
    prob = entrenar_predecir(df, features, cfg["velas"])
    ultima = df.iloc[-1]
    precio = float(ultima["close"].item() if hasattr(ultima["close"],'item') else ultima["close"])
    atr    = float(ultima["atr"].item()   if hasattr(ultima["atr"],  'item') else ultima["atr"])
    vwap   = float(ultima["vwap"].item()  if hasattr(ultima["vwap"], 'item') else ultima["vwap"])
    bar    = int(ultima["bar_of_day"])
    reg_v  = int(reg.iloc[-1])
    sl_mc = tp_mc = None
    if prob >= PROB_MINIMA:
        sl_mc = precio-atr*ATR_SL_MULT; tp_mc = precio+atr*ATR_TP_MULT
    elif prob <= 1-PROB_MINIMA:
        sl_mc = precio+atr*ATR_SL_MULT; tp_mc = precio-atr*ATR_TP_MULT
    target = monte_carlo(df, cfg["velas"], sl=sl_mc, tp=tp_mc)
    return prob, precio, atr, vwap, bar, reg_v, target, cfg, df

# ══════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════

st.markdown("""
<div style='text-align:center; padding: 10px 0 4px 0;'>
  <span style='color:#00d4ff; font-family:Courier New; font-size:1.4rem; font-weight:bold;'>
    ⚡ QUANTUM SIGNAL PREDICTOR v7.0
  </span><br>
  <span style='color:#4a5568; font-family:Courier New; font-size:0.75rem;'>
    Motor v2 · AdaBoost · Monte Carlo 10K · Noticias en tiempo real
  </span>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("### ⚙️ Configuracion")
    ticker = st.text_input("Ticker", value="SPY").strip().upper()
    unidad = st.radio("Unidad de horizonte", ["min", "dias"], horizontal=True)
    valor  = st.number_input(
        "Horizonte" + (" (minutos)" if unidad=="min" else " (dias)"),
        min_value=1, max_value=252,
        value=15 if unidad=="min" else 5, step=1)
    analizar_btn = st.button("⚡ ANALIZAR", type="primary", use_container_width=True)
    st.markdown("---")
    auto_refresh = st.checkbox("Auto-refresh cada 60 seg", value=False)
    st.markdown("---")
    st.markdown("""
    <div style='color:#4a5568; font-size:0.72rem; font-family:Courier New;'>
    <b style='color:#8892a4;'>Senales:</b><br>
    🟢 LONG — prob >= 70%<br>
    🔴 SHORT — prob <= 30%<br>
    🟡 NEUTRAL — entre 30-70%<br><br>
    <b style='color:#8892a4;'>Regimen:</b><br>
    0 = Alcista fuerte<br>
    1 = Bajista fuerte<br>
    2 = Lateral<br>
    3 = Alta volatilidad
    </div>
    """, unsafe_allow_html=True)

# ── MAIN ──
col_signal, col_news = st.columns([3, 2])

with col_signal:
    if analizar_btn or (auto_refresh and "ultima_corrida" in st.session_state):
        if not ticker:
            st.warning("Ingresa un ticker.")
        else:
            with st.spinner(f"Analizando {ticker}..."):
                try:
                    prob, precio, atr, vwap, bar, reg_v, target, cfg, df = analizar(ticker, unidad, int(valor))
                    st.session_state["ultima_corrida"] = datetime.now()
                    st.session_state["ultima_senal"] = prob

                    # ── SENAL PRINCIPAL ──
                    reg_nombres = {0:"ALCISTA FUERTE", 1:"BAJISTA FUERTE", 2:"LATERAL", 3:"ALTA VOL"}
                    reg_colores = {0:"#00ff88", 1:"#ff3366", 2:"#ffcc00", 3:"#ff8800"}

                    if prob >= PROB_MINIMA:
                        senal,accion,color,css = "LONG","COMPRAR","#00ff88","signal-long"
                        sl = precio-atr*ATR_SL_MULT; tp_price = precio+atr*ATR_TP_MULT; pd_ = prob
                    elif prob <= 1-PROB_MINIMA:
                        senal,accion,color,css = "SHORT","VENDER / SHORT","#ff3366","signal-short"
                        sl = precio+atr*ATR_SL_MULT; tp_price = precio-atr*ATR_TP_MULT; pd_ = 1-prob
                    else:
                        senal,accion,color,css = "NEUTRAL","NO OPERAR","#ffcc00","signal-neutral"
                        sl = tp_price = 0; pd_ = abs(prob-0.5)*2

                    barra = "█"*int(pd_*20) + "░"*(20-int(pd_*20))
                    riesgo    = abs(precio-sl) if sl else 0
                    beneficio = abs(tp_price-precio) if tp_price else 0

                    st.markdown(f"""
                    <div style='background:#0d1528; border-radius:8px; padding:16px; margin-bottom:12px;'>
                      <div class='{css}'>{senal}</div>
                      <div style='color:{color}; font-family:Courier New; font-size:1rem;'>{accion}</div>
                      <div style='color:#8892a4; font-family:Courier New; font-size:0.8rem; margin-top:6px;'>
                        Probabilidad: {pd_*100:.1f}%&nbsp;&nbsp;[{barra}]
                      </div>
                      <div style='color:#4a5568; font-family:Courier New; font-size:0.7rem; margin-top:4px;'>
                        Modo: {cfg["label"]} | Vela: {cfg["intervalo"]} | Historial: {cfg["periodo"]}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── METRICAS ──
                    c1,c2,c3,c4 = st.columns(4)
                    c1.metric("Precio", f"${precio:.2f}")
                    c2.metric("VWAP", f"${vwap:.2f}", "↑ arriba" if precio>vwap else "↓ abajo")
                    c3.metric("ATR", f"${atr:.2f}")
                    c4.metric("Regimen", reg_nombres.get(reg_v,"—"),
                              delta=None)

                    if sl and tp_price:
                        c5,c6,c7 = st.columns(3)
                        c5.metric("Stop Loss", f"${sl:.2f}", f"-${riesgo:.2f}")
                        c6.metric("Take Profit", f"${tp_price:.2f}", f"+${beneficio:.2f}")
                        c7.metric("Ratio R/B", f"1:{beneficio/riesgo:.1f}" if riesgo>0 else "—")

                    # ── MONTE CARLO ──
                    if target:
                        st.markdown("#### 🎲 Monte Carlo")
                        mc1,mc2,mc3,mc4 = st.columns(4)
                        ps = target["prob_sub"]
                        ds = "SUBE" if ps>=0.55 else ("BAJA" if ps<=0.45 else "LATERAL")
                        mc1.metric("Target MC", f"${target['target']:.2f}", f"{target['ret_med']:+.2f}%")
                        mc2.metric("Rango MC", f"${target['bear']:.2f} — ${target['bull']:.2f}")
                        mc3.metric("Prob. suba", f"{ps*100:.0f}% {ds}")
                        if target.get("prob_tp") is not None:
                            ptp = target["prob_tp"]
                            ok = "OK" if ptp>=0.55 else ("NO" if ptp<=0.40 else "NEUTRO")
                            mc4.metric("Prob. TP/SL", f"{ptp*100:.0f}% {ok}")

                    # ── SIZING ──
                    if riesgo > 0:
                        st.markdown("#### 💰 Sizing (riesgo 1%)")
                        sz_cols = st.columns(4)
                        for i, cap in enumerate([5000,10000,25000,50000]):
                            size = int((cap*0.01)/riesgo)
                            sz_cols[i].metric(
                                f"Capital ${cap:,}",
                                f"{size} acc",
                                f"+${size*beneficio:.0f}" if beneficio else None
                            )

                    # ── ADVERTENCIAS ──
                    advs = []
                    if reg_v == 3: advs.append("⚠️ ALTA VOLATILIDAD — reduce el tamaño")
                    if bar < 6:    advs.append("⚠️ PRIMERA MEDIA HORA — evitar operar")
                    if bar > 70:   advs.append("⚠️ ULTIMA MEDIA HORA — evitar operar")
                    if reg_v==1 and senal=="LONG":  advs.append("⚠️ LONG en tendencia BAJISTA")
                    if reg_v==0 and senal=="SHORT": advs.append("⚠️ SHORT en tendencia ALCISTA")
                    if advs:
                        for adv in advs:
                            st.warning(adv)

                    # ── GRAFICO PRECIO ──
                    with st.expander("📈 Ver grafico de precio"):
                        chart_df = df[["close"]].tail(100).copy()
                        st.line_chart(chart_df)

                    st.caption(f"Actualizado: {datetime.now().strftime('%H:%M:%S')} UY")

                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.markdown("""
        <div style='background:#0d1528; border-radius:8px; padding:32px; text-align:center; margin-top:20px;'>
          <div style='color:#4a5568; font-family:Courier New; font-size:1rem;'>
            Ingresa un ticker y presiona<br>
            <span style='color:#00d4ff; font-size:1.2rem;'>⚡ ANALIZAR</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── NOTICIAS ──
with col_news:
    st.markdown("#### 📰 Noticias — Impacto en Mercado")
    with st.spinner("Cargando noticias..."):
        noticias = obtener_noticias()

    for n in noticias:
        prefix = "🇺🇸 " if n["trump"] else ""
        css_class = f"news-{n['css']}"
        st.markdown(f"""
        <div style='background:#0d1528; border-radius:6px; padding:8px 10px; margin-bottom:6px;'>
          <span class='{css_class}' style='font-family:Courier New; font-size:0.75rem;'>
            {n['impacto']}
          </span>
          <span style='color:#4a5568; font-family:Courier New; font-size:0.65rem;'>
            &nbsp;{n['tiempo']} · {n['fuente']}
          </span><br>
          <span style='color:#c9d1d9; font-family:Courier New; font-size:0.75rem;'>
            {prefix}{n['titulo']}
          </span>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🔄 Actualizar noticias"):
        st.cache_data.clear()
        st.rerun()

# ── AUTO REFRESH ──
if auto_refresh:
    time.sleep(60)
    st.rerun()
