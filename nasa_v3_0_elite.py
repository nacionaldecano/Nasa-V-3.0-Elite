
import math
import warnings
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

st.set_page_config(page_title="NASA v3.0 ELITE", layout="wide", initial_sidebar_state="expanded")

DEFAULT_PERIOD = "1y"
DEFAULT_INTERVAL = "1d"
DEFAULT_MC_PATHS = 10000
DEFAULT_MC_DAYS = 5

@dataclass
class TechnicalSummary:
    price: float
    sma20: float
    sma50: float
    sma200: float
    ema12: float
    ema26: float
    rsi14: float
    macd: float
    macd_signal: float
    macd_hist: float
    bb_upper: float
    bb_mid: float
    bb_lower: float
    atr14: float
    daily_return: float
    volatility20: float
    trend_signal: str
    momentum_signal: str
    support: float
    resistance: float

@dataclass
class OptionsSummary:
    expiration: str
    max_pain: float
    put_call_oi_ratio: float
    call_oi_total: int
    put_oi_total: int
    expected_move: float
    implied_move_pct: float
    atm_strike: float
    call_wall: float
    put_wall: float
    options_available: bool
    notes: str

@dataclass
class MonteCarloSummary:
    expected_close: float
    expected_high: float
    expected_low: float
    p10_close: float
    p50_close: float
    p90_close: float
    bullish_prob: float
    bearish_prob: float
    neutral_prob: float

@dataclass
class EliteSummary:
    ticker: str
    direction: str
    confidence: float
    last_price: float
    intraday_high_est: float
    intraday_low_est: float
    daily_close_est: float
    weekly_high_est: float
    weekly_low_est: float
    weekly_close_est: float
    strategy: str
    options_mode: str
    notes: str

def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default

def annualize_volatility(std_daily: float):
    if pd.isna(std_daily):
        return np.nan
    return std_daily * np.sqrt(252)

def nearest_value(values, target):
    if len(values) == 0:
        return np.nan
    arr = np.array(values, dtype=float)
    idx = np.abs(arr - target).argmin()
    return float(arr[idx])

def classify_confidence(score):
    if score >= 0.75:
        return 82.0
    if score >= 0.55:
        return 68.0
    if score >= 0.45:
        return 57.0
    return 51.0

def download_market_data(ticker: str, period=DEFAULT_PERIOD, interval=DEFAULT_INTERVAL) -> pd.DataFrame:
    data = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False, threads=False)
    if data is None or len(data) == 0:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] for c in data.columns]
    data = data.rename(columns=str.title)
    expected = ["Open", "High", "Low", "Close", "Volume"]
    for c in expected:
        if c not in data.columns:
            data[c] = np.nan
    data.dropna(subset=["Close"], inplace=True)
    return data

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["SMA20"] = out["Close"].rolling(20).mean()
    out["SMA50"] = out["Close"].rolling(50).mean()
    out["SMA200"] = out["Close"].rolling(200).mean()
    out["EMA12"] = out["Close"].ewm(span=12, adjust=False).mean()
    out["EMA26"] = out["Close"].ewm(span=26, adjust=False).mean()

    delta = out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI14"] = 100 - (100 / (1 + rs))

    out["MACD"] = out["EMA12"] - out["EMA26"]
    out["MACD_SIGNAL"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_HIST"] = out["MACD"] - out["MACD_SIGNAL"]

    out["BB_MID"] = out["Close"].rolling(20).mean()
    std20 = out["Close"].rolling(20).std()
    out["BB_UPPER"] = out["BB_MID"] + 2 * std20
    out["BB_LOWER"] = out["BB_MID"] - 2 * std20

    high_low = out["High"] - out["Low"]
    high_close = (out["High"] - out["Close"].shift()).abs()
    low_close = (out["Low"] - out["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    out["ATR14"] = tr.rolling(14).mean()

    out["RETURN"] = out["Close"].pct_change()
    out["VOL20"] = out["RETURN"].rolling(20).std()
    out["SUPPORT20"] = out["Low"].rolling(20).min()
    out["RESIST20"] = out["High"].rolling(20).max()
    return out

def summarize_technicals(df: pd.DataFrame) -> TechnicalSummary:
    last = df.iloc[-1]
    trend_score = 0
    if last["Close"] > last["SMA20"]:
        trend_score += 1
    if last["Close"] > last["SMA50"]:
        trend_score += 1
    if pd.notna(last["SMA200"]) and last["Close"] > last["SMA200"]:
        trend_score += 1
    trend_signal = "Bullish" if trend_score >= 2 else "Neutral" if trend_score == 1 else "Bearish"

    momentum_score = 0
    if pd.notna(last["RSI14"]) and last["RSI14"] > 55:
        momentum_score += 1
    elif pd.notna(last["RSI14"]) and last["RSI14"] < 45:
        momentum_score -= 1
    if pd.notna(last["MACD_HIST"]) and last["MACD_HIST"] > 0:
        momentum_score += 1
    elif pd.notna(last["MACD_HIST"]) and last["MACD_HIST"] < 0:
        momentum_score -= 1
    momentum_signal = "Bullish" if momentum_score >= 1 else "Bearish" if momentum_score <= -1 else "Neutral"

    return TechnicalSummary(
        price=safe_float(last["Close"]),
        sma20=safe_float(last["SMA20"]),
        sma50=safe_float(last["SMA50"]),
        sma200=safe_float(last["SMA200"]),
        ema12=safe_float(last["EMA12"]),
        ema26=safe_float(last["EMA26"]),
        rsi14=safe_float(last["RSI14"]),
        macd=safe_float(last["MACD"]),
        macd_signal=safe_float(last["MACD_SIGNAL"]),
        macd_hist=safe_float(last["MACD_HIST"]),
        bb_upper=safe_float(last["BB_UPPER"]),
        bb_mid=safe_float(last["BB_MID"]),
        bb_lower=safe_float(last["BB_LOWER"]),
        atr14=safe_float(last["ATR14"]),
        daily_return=safe_float(last["RETURN"]),
        volatility20=safe_float(annualize_volatility(last["VOL20"])),
        trend_signal=trend_signal,
        momentum_signal=momentum_signal,
        support=safe_float(last["SUPPORT20"]),
        resistance=safe_float(last["RESIST20"]),
    )

def get_option_expirations(ticker: str):
    try:
        tk = yf.Ticker(ticker)
        exps = tk.options
        return list(exps) if exps else []
    except Exception:
        return []

def load_option_chain(ticker: str, expiration: str):
    try:
        tk = yf.Ticker(ticker)
        chain = tk.option_chain(expiration)
        return chain.calls.copy(), chain.puts.copy()
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

def compute_max_pain(calls: pd.DataFrame, puts: pd.DataFrame):
    if calls.empty or puts.empty:
        return np.nan
    call_strikes = calls["strike"].dropna().unique().tolist()
    put_strikes = puts["strike"].dropna().unique().tolist()
    all_strikes = sorted(set(call_strikes + put_strikes))
    if len(all_strikes) == 0:
        return np.nan
    pain_by_strike = []
    for s in all_strikes:
        total_pain = 0.0
        for _, row in calls.iterrows():
            strike = safe_float(row.get("strike"))
            oi = safe_float(row.get("openInterest"), 0.0)
            total_pain += max(0.0, s - strike) * oi
        for _, row in puts.iterrows():
            strike = safe_float(row.get("strike"))
            oi = safe_float(row.get("openInterest"), 0.0)
            total_pain += max(0.0, strike - s) * oi
        pain_by_strike.append((s, total_pain))
    pain_df = pd.DataFrame(pain_by_strike, columns=["strike", "pain"])
    return safe_float(pain_df.loc[pain_df["pain"].idxmin(), "strike"])

def compute_expected_move(calls: pd.DataFrame, puts: pd.DataFrame, spot: float):
    if calls.empty or puts.empty or pd.isna(spot):
        return np.nan, np.nan, np.nan
    strikes = sorted(set(calls["strike"].dropna().tolist()) & set(puts["strike"].dropna().tolist()))
    if len(strikes) == 0:
        return np.nan, np.nan, np.nan
    atm = nearest_value(strikes, spot)
    call_atm = calls.loc[calls["strike"] == atm]
    put_atm = puts.loc[puts["strike"] == atm]
    if call_atm.empty or put_atm.empty:
        return atm, np.nan, np.nan
    call_mid = safe_float((call_atm["bid"].fillna(0) + call_atm["ask"].fillna(0)).iloc[0] / 2.0)
    put_mid = safe_float((put_atm["bid"].fillna(0) + put_atm["ask"].fillna(0)).iloc[0] / 2.0)
    if (call_mid == 0 and put_mid == 0) or pd.isna(call_mid) or pd.isna(put_mid):
        call_last = safe_float(call_atm["lastPrice"].iloc[0])
        put_last = safe_float(put_atm["lastPrice"].iloc[0])
        expected_move = call_last + put_last
    else:
        expected_move = call_mid + put_mid
    implied_move_pct = (expected_move / spot) * 100 if spot else np.nan
    return atm, expected_move, implied_move_pct

def summarize_options(ticker: str, spot: float) -> OptionsSummary:
    exps = get_option_expirations(ticker)
    if not exps:
        return OptionsSummary("N/A", np.nan, np.nan, 0, 0, np.nan, np.nan, np.nan, np.nan, np.nan, False, "No listed options or unavailable from source.")
    expiration = exps[0]
    calls, puts = load_option_chain(ticker, expiration)
    if calls.empty or puts.empty:
        return OptionsSummary(expiration, np.nan, np.nan, 0, 0, np.nan, np.nan, np.nan, np.nan, np.nan, False, "Option chain unavailable for selected expiration.")
    calls["openInterest"] = pd.to_numeric(calls["openInterest"], errors="coerce").fillna(0)
    puts["openInterest"] = pd.to_numeric(puts["openInterest"], errors="coerce").fillna(0)
    call_oi_total = int(calls["openInterest"].sum())
    put_oi_total = int(puts["openInterest"].sum())
    put_call_oi_ratio = safe_float(put_oi_total / call_oi_total) if call_oi_total > 0 else np.nan
    max_pain = compute_max_pain(calls, puts)
    atm_strike, expected_move, implied_move_pct = compute_expected_move(calls, puts, spot)
    call_wall = safe_float(calls.loc[calls["openInterest"].idxmax(), "strike"]) if not calls.empty else np.nan
    put_wall = safe_float(puts.loc[puts["openInterest"].idxmax(), "strike"]) if not puts.empty else np.nan
    return OptionsSummary(expiration, max_pain, put_call_oi_ratio, call_oi_total, put_oi_total, expected_move, implied_move_pct, atm_strike, call_wall, put_wall, True, "Best-effort public options analysis from Yahoo source.")

def compute_sentiment_proxy(df: pd.DataFrame, technicals: TechnicalSummary):
    last_5 = df["Close"].pct_change(5).iloc[-1] if len(df) > 6 else np.nan
    last_20 = df["Close"].pct_change(20).iloc[-1] if len(df) > 21 else np.nan
    score = 0.5
    if pd.notna(last_5):
        score += np.clip(last_5, -0.1, 0.1)
    if pd.notna(last_20):
        score += np.clip(last_20 / 2, -0.1, 0.1)
    if pd.notna(technicals.rsi14):
        if technicals.rsi14 > 60:
            score += 0.05
        elif technicals.rsi14 < 40:
            score -= 0.05
    score = float(np.clip(score, 0.0, 1.0))
    label = "Positive" if score > 0.58 else "Negative" if score < 0.42 else "Neutral"
    return score, label

def monte_carlo_paths(df: pd.DataFrame, days=DEFAULT_MC_DAYS, n_paths=DEFAULT_MC_PATHS):
    returns = df["Close"].pct_change().dropna()
    if len(returns) < 30:
        return None
    mu = returns.mean()
    sigma = returns.std()
    last_price = safe_float(df["Close"].iloc[-1])
    if pd.isna(last_price):
        return None
    all_paths = []
    for _ in range(n_paths):
        price = last_price
        path = [price]
        for _ in range(days):
            shock = np.random.normal(mu, sigma)
            price = price * (1 + shock)
            path.append(price)
        all_paths.append(path)
    sim_df = pd.DataFrame(all_paths).T
    sim_df.index = range(0, days + 1)
    return sim_df

def summarize_monte_carlo(sim_df: pd.DataFrame, last_price: float) -> MonteCarloSummary:
    final_prices = sim_df.iloc[-1]
    expected_close = safe_float(final_prices.mean())
    p10 = safe_float(final_prices.quantile(0.10))
    p50 = safe_float(final_prices.quantile(0.50))
    p90 = safe_float(final_prices.quantile(0.90))
    expected_high = safe_float(sim_df.max().max())
    expected_low = safe_float(sim_df.min().min())
    up_threshold = last_price * 1.01
    down_threshold = last_price * 0.99
    bullish_prob = float((final_prices > up_threshold).mean())
    bearish_prob = float((final_prices < down_threshold).mean())
    neutral_prob = float(1 - bullish_prob - bearish_prob)
    return MonteCarloSummary(expected_close, expected_high, expected_low, p10, p50, p90, bullish_prob, bearish_prob, neutral_prob)

def elite_decision_engine(ticker: str, df: pd.DataFrame, tech: TechnicalSummary, opt: OptionsSummary, mc: MonteCarloSummary, sentiment_score: float, sentiment_label: str) -> EliteSummary:
    score = 0.5
    if tech.trend_signal == "Bullish":
        score += 0.12
    elif tech.trend_signal == "Bearish":
        score -= 0.12
    if tech.momentum_signal == "Bullish":
        score += 0.08
    elif tech.momentum_signal == "Bearish":
        score -= 0.08
    if pd.notna(tech.rsi14):
        if tech.rsi14 > 60:
            score += 0.05
        elif tech.rsi14 < 40:
            score -= 0.05
    if pd.notna(tech.macd_hist):
        if tech.macd_hist > 0:
            score += 0.04
        elif tech.macd_hist < 0:
            score -= 0.04
    score += (sentiment_score - 0.5) * 0.20

    options_mode = "Unavailable"
    notes = []
    if opt.options_available:
        options_mode = "Enabled"
        if pd.notna(opt.max_pain):
            if tech.price > opt.max_pain:
                score += 0.01
            elif tech.price < opt.max_pain:
                score -= 0.01
        if pd.notna(opt.put_call_oi_ratio):
            if opt.put_call_oi_ratio > 1.2:
                notes.append("Heavy put positioning / defensive tone.")
                score -= 0.04
            elif opt.put_call_oi_ratio < 0.8:
                notes.append("Call-heavy positioning.")
                score += 0.04
        if pd.notna(opt.call_wall) and pd.notna(opt.put_wall):
            if tech.price >= opt.call_wall * 0.99:
                notes.append("Price near call wall / resistance zone.")
                score -= 0.03
            if tech.price <= opt.put_wall * 1.01:
                notes.append("Price near put wall / support zone.")
                score += 0.03

    score += (mc.bullish_prob - mc.bearish_prob) * 0.20
    score = float(np.clip(score, 0.0, 1.0))

    if score > 0.57:
        direction = "Bullish"
        strategy = "Favor pullback buys / trend continuation."
    elif score < 0.43:
        direction = "Bearish"
        strategy = "Favor rallies to fade / defensive posture."
    else:
        direction = "Neutral"
        strategy = "Range / wait for confirmation."

    confidence = classify_confidence(score)
    intraday_range = tech.atr14 if pd.notna(tech.atr14) else tech.price * 0.02
    if pd.notna(opt.expected_move):
        intraday_range = min(max(intraday_range, opt.expected_move * 0.35), opt.expected_move)
    intraday_high = tech.price + intraday_range / 2
    intraday_low = tech.price - intraday_range / 2
    notes.append(f"Sentiment proxy: {sentiment_label}.")
    if pd.notna(opt.expected_move):
        notes.append(f"Expected move approx: {opt.expected_move:.2f} ({opt.implied_move_pct:.2f}%).")
    return EliteSummary(
        ticker, direction, confidence, tech.price,
        safe_float(intraday_high), safe_float(intraday_low),
        safe_float(mc.expected_close), safe_float(mc.expected_high),
        safe_float(mc.expected_low), safe_float(mc.expected_close),
        strategy, options_mode, " ".join(notes)
    )

def run_nasa_elite(ticker: str, period: str, interval: str, mc_days: int, mc_paths: int):
    raw = download_market_data(ticker, period=period, interval=interval)
    if raw.empty or len(raw) < 60:
        return {"ok": False, "ticker": ticker, "error": "Insufficient or unavailable market data."}
    df = add_indicators(raw)
    tech = summarize_technicals(df)
    opt = summarize_options(ticker, tech.price)
    sentiment_score, sentiment_label = compute_sentiment_proxy(df, tech)
    sim_df = monte_carlo_paths(df, days=mc_days, n_paths=mc_paths)
    if sim_df is None:
        return {"ok": False, "ticker": ticker, "error": "Monte Carlo engine could not initialize."}
    mc = summarize_monte_carlo(sim_df, tech.price)
    elite = elite_decision_engine(ticker, df, tech, opt, mc, sentiment_score, sentiment_label)
    return {
        "ok": True, "ticker": ticker, "raw": raw, "df": df,
        "technicals": tech, "options": opt, "sentiment_score": sentiment_score,
        "sentiment_label": sentiment_label, "sim_df": sim_df, "mc": mc, "elite": elite
    }

def metric_row_4(a1, a2, a3, a4, labels):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(labels[0], a1)
    c2.metric(labels[1], a2)
    c3.metric(labels[2], a3)
    c4.metric(labels[3], a4)

def format_num(x):
    if pd.isna(x):
        return "N/A"
    return f"{x:,.2f}"

st.title("🚀 NASA v3.0 ELITE")
st.caption("Versión completa - multi-ticker, técnicos, opciones, Monte Carlo y resumen ELITE.")

with st.sidebar:
    st.header("Configuración")
    tickers_input = st.text_input("Tickers (separados por coma)", value="QQQ,TQQQ,NVDA,MSFT")
    period = st.selectbox("Período histórico", ["6mo", "1y", "2y", "5y"], index=1)
    interval = st.selectbox("Intervalo", ["1d", "1h"], index=0)
    mc_days = st.number_input("Horizonte Monte Carlo (días)", min_value=1, max_value=30, value=5, step=1)
    mc_paths = st.number_input("Trayectorias Monte Carlo", min_value=1000, max_value=25000, value=10000, step=1000)

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
if not tickers:
    st.warning("Ingresá al menos un ticker.")
    st.stop()

all_summary_rows = []

for ticker in tickers:
    st.markdown("---")
    st.header(ticker)
    result = run_nasa_elite(ticker=ticker, period=period, interval=interval, mc_days=int(mc_days), mc_paths=int(mc_paths))
    if not result["ok"]:
        st.error(f"{ticker}: {result['error']}")
        continue

    df = result["df"]
    tech = result["technicals"]
    opt = result["options"]
    mc = result["mc"]
    elite = result["elite"]

    metric_row_4(format_num(elite.last_price), elite.direction, f"{elite.confidence:.0f}%", elite.options_mode, ["Precio", "Dirección", "Confianza", "Modo Opciones"])

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Resumen ELITE", "Técnicos", "Opciones", "Monte Carlo", "Datos"])

    with tab1:
        summary_df = pd.DataFrame([{
            "Ticker": elite.ticker,
            "Dirección": elite.direction,
            "Confianza %": elite.confidence,
            "Precio Actual": elite.last_price,
            "Máximo Intradía Est.": elite.intraday_high_est,
            "Mínimo Intradía Est.": elite.intraday_low_est,
            "Cierre Diario Est.": elite.daily_close_est,
            "Máximo Semanal Est.": elite.weekly_high_est,
            "Mínimo Semanal Est.": elite.weekly_low_est,
            "Cierre Semanal Est.": elite.weekly_close_est,
            "Estrategia": elite.strategy,
            "Notas": elite.notes
        }])
        st.dataframe(summary_df, use_container_width=True)
        key_levels = pd.DataFrame([{
            "Soporte": tech.support,
            "Resistencia": tech.resistance,
            "BB Inferior": tech.bb_lower,
            "BB Media": tech.bb_mid,
            "BB Superior": tech.bb_upper,
            "ATR14": tech.atr14,
            "Max Pain": opt.max_pain,
            "Call Wall": opt.call_wall,
            "Put Wall": opt.put_wall,
        }])
        st.dataframe(key_levels, use_container_width=True)
        st.info(elite.strategy)

    with tab2:
        tech_df = pd.DataFrame([{
            "Precio": tech.price, "SMA20": tech.sma20, "SMA50": tech.sma50, "SMA200": tech.sma200,
            "EMA12": tech.ema12, "EMA26": tech.ema26, "RSI14": tech.rsi14, "MACD": tech.macd,
            "MACD Signal": tech.macd_signal, "MACD Hist": tech.macd_hist, "ATR14": tech.atr14,
            "Volatilidad 20d Anualizada": tech.volatility20, "Trend Signal": tech.trend_signal,
            "Momentum Signal": tech.momentum_signal
        }])
        st.dataframe(tech_df, use_container_width=True)
        chart_cols = ["Close", "SMA20", "SMA50", "SMA200"]
        st.line_chart(df[chart_cols])
        st.line_chart(df[["Close", "BB_UPPER", "BB_MID", "BB_LOWER"]])
        st.line_chart(df[["MACD", "MACD_SIGNAL", "MACD_HIST"]])

    with tab3:
        opt_df = pd.DataFrame([{
            "Expiration": opt.expiration, "Options Available": opt.options_available,
            "Max Pain": opt.max_pain, "Put/Call OI Ratio": opt.put_call_oi_ratio,
            "Call OI Total": opt.call_oi_total, "Put OI Total": opt.put_oi_total,
            "Expected Move": opt.expected_move, "Implied Move %": opt.implied_move_pct,
            "ATM Strike": opt.atm_strike, "Call Wall": opt.call_wall, "Put Wall": opt.put_wall,
            "Notes": opt.notes
        }])
        st.dataframe(opt_df, use_container_width=True)
        if opt.options_available and pd.notna(opt.expected_move):
            em_up = tech.price + opt.expected_move
            em_down = tech.price - opt.expected_move
            st.write(f"Rango esperado por opciones: **{format_num(em_down)} – {format_num(em_up)}**")

    with tab4:
        mc_df = pd.DataFrame([{
            "Expected Close": mc.expected_close, "Expected High": mc.expected_high,
            "Expected Low": mc.expected_low, "P10 Close": mc.p10_close, "P50 Close": mc.p50_close,
            "P90 Close": mc.p90_close, "Bullish Probability": mc.bullish_prob,
            "Bearish Probability": mc.bearish_prob, "Neutral Probability": mc.neutral_prob,
        }])
        st.dataframe(mc_df, use_container_width=True)
        st.line_chart(result["sim_df"].iloc[:, : min(200, result["sim_df"].shape[1])])

    with tab5:
        st.dataframe(df.tail(50), use_container_width=True)

    all_summary_rows.append({
        "Ticker": elite.ticker, "Direction": elite.direction, "Confidence %": elite.confidence,
        "Price": elite.last_price, "Intraday High Est.": elite.intraday_high_est,
        "Intraday Low Est.": elite.intraday_low_est, "Daily Close Est.": elite.daily_close_est,
        "Weekly High Est.": elite.weekly_high_est, "Weekly Low Est.": elite.weekly_low_est,
        "Weekly Close Est.": elite.weekly_close_est, "Strategy": elite.strategy,
        "Options Mode": elite.options_mode,
    })

if all_summary_rows:
    st.markdown("---")
    st.header("Resumen global NASA v3.0 ELITE")
    global_df = pd.DataFrame(all_summary_rows)
    st.dataframe(global_df, use_container_width=True)
    csv = global_df.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar resumen CSV", data=csv, file_name=f"nasa_v3_elite_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
    st.success("Script NASA v3.0 ELITE cargado.")
