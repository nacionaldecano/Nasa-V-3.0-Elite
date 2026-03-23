
import math
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

st.set_page_config(page_title="NASA v4.0 ELITE", layout="wide", initial_sidebar_state="expanded")

DEFAULT_PERIOD   = "1y"
DEFAULT_INTERVAL = "1d"
DEFAULT_MC_PATHS = 10000
DEFAULT_MC_DAYS  = 5

# ─────────────────────────────────────────────
# DATACLASSES
# ─────────────────────────────────────────────

@dataclass
class TechnicalSummary:
    price: float
    sma20: float; sma50: float; sma200: float
    ema12: float; ema26: float
    rsi14: float
    macd: float; macd_signal: float; macd_hist: float
    bb_upper: float; bb_mid: float; bb_lower: float
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
    # NEW
    gex: float                    # Gamma Exposure neto
    gamma_flip: float             # Nivel donde GEX cambia de signo
    dealer_position: str          # "Long Gamma" / "Short Gamma" / "Neutral"
    vol_skew_25d: float           # IV skew 25-delta put vs call
    calls_df: object              # DataFrame completo de calls
    puts_df: object               # DataFrame completo de puts
    options_available: bool
    notes: str

@dataclass
class MonteCarloSummary:
    expected_close: float
    expected_high: float
    expected_low: float
    p10_close: float; p50_close: float; p90_close: float
    bullish_prob: float; bearish_prob: float; neutral_prob: float

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

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def safe_float(x, default=np.nan):
    try:
        if pd.isna(x): return default
        return float(x)
    except Exception:
        return default

def annualize_volatility(std_daily: float):
    return std_daily * np.sqrt(252) if pd.notna(std_daily) else np.nan

def nearest_value(values, target):
    if len(values) == 0: return np.nan
    arr = np.array(values, dtype=float)
    return float(arr[np.abs(arr - target).argmin()])

def classify_confidence(score):
    if score >= 0.75: return 82.0
    if score >= 0.55: return 68.0
    if score >= 0.45: return 57.0
    return 51.0

# ─────────────────────────────────────────────
# MARKET DATA
# ─────────────────────────────────────────────

def download_market_data(ticker: str, period=DEFAULT_PERIOD, interval=DEFAULT_INTERVAL) -> pd.DataFrame:
    data = yf.download(ticker, period=period, interval=interval,
                       auto_adjust=True, progress=False, threads=False)
    if data is None or len(data) == 0:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] for c in data.columns]
    data = data.rename(columns=str.title)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c not in data.columns:
            data[c] = np.nan
    data.dropna(subset=["Close"], inplace=True)
    return data

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["SMA20"]  = out["Close"].rolling(20).mean()
    out["SMA50"]  = out["Close"].rolling(50).mean()
    out["SMA200"] = out["Close"].rolling(200).mean()
    out["EMA12"]  = out["Close"].ewm(span=12, adjust=False).mean()
    out["EMA26"]  = out["Close"].ewm(span=26, adjust=False).mean()

    delta    = out["Close"].diff()
    avg_gain = delta.clip(lower=0).rolling(14).mean()
    avg_loss = (-delta.clip(upper=0)).rolling(14).mean()
    out["RSI14"] = 100 - (100 / (1 + avg_gain / avg_loss.replace(0, np.nan)))

    out["MACD"]        = out["EMA12"] - out["EMA26"]
    out["MACD_SIGNAL"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_HIST"]   = out["MACD"] - out["MACD_SIGNAL"]

    out["BB_MID"]   = out["Close"].rolling(20).mean()
    std20           = out["Close"].rolling(20).std()
    out["BB_UPPER"] = out["BB_MID"] + 2 * std20
    out["BB_LOWER"] = out["BB_MID"] - 2 * std20

    hl  = out["High"] - out["Low"]
    hc  = (out["High"] - out["Close"].shift()).abs()
    lc  = (out["Low"]  - out["Close"].shift()).abs()
    out["ATR14"]    = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    out["RETURN"]    = out["Close"].pct_change()
    out["VOL20"]     = out["RETURN"].rolling(20).std()
    out["SUPPORT20"] = out["Low"].rolling(20).min()
    out["RESIST20"]  = out["High"].rolling(20).max()
    return out

def summarize_technicals(df: pd.DataFrame) -> TechnicalSummary:
    last = df.iloc[-1]
    ts = sum([last["Close"] > last["SMA20"],
              last["Close"] > last["SMA50"],
              pd.notna(last["SMA200"]) and last["Close"] > last["SMA200"]])
    trend_signal = "Bullish" if ts >= 2 else "Neutral" if ts == 1 else "Bearish"

    ms = 0
    if pd.notna(last["RSI14"]):
        ms += 1 if last["RSI14"] > 55 else (-1 if last["RSI14"] < 45 else 0)
    if pd.notna(last["MACD_HIST"]):
        ms += 1 if last["MACD_HIST"] > 0 else -1
    momentum_signal = "Bullish" if ms >= 1 else "Bearish" if ms <= -1 else "Neutral"

    return TechnicalSummary(
        price=safe_float(last["Close"]),
        sma20=safe_float(last["SMA20"]), sma50=safe_float(last["SMA50"]),
        sma200=safe_float(last["SMA200"]), ema12=safe_float(last["EMA12"]),
        ema26=safe_float(last["EMA26"]), rsi14=safe_float(last["RSI14"]),
        macd=safe_float(last["MACD"]), macd_signal=safe_float(last["MACD_SIGNAL"]),
        macd_hist=safe_float(last["MACD_HIST"]), bb_upper=safe_float(last["BB_UPPER"]),
        bb_mid=safe_float(last["BB_MID"]), bb_lower=safe_float(last["BB_LOWER"]),
        atr14=safe_float(last["ATR14"]), daily_return=safe_float(last["RETURN"]),
        volatility20=safe_float(annualize_volatility(last["VOL20"])),
        trend_signal=trend_signal, momentum_signal=momentum_signal,
        support=safe_float(last["SUPPORT20"]), resistance=safe_float(last["RESIST20"]),
    )

# ─────────────────────────────────────────────
# BLACK-SCHOLES HELPERS (para GEX y skew)
# ─────────────────────────────────────────────

def bs_gamma(S, K, T, r, sigma):
    """Black-Scholes gamma."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        return math.exp(-0.5 * d1**2) / (S * sigma * math.sqrt(2 * math.pi * T))
    except Exception:
        return 0.0

def bs_delta_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    try:
        from scipy.stats import norm
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        return float(norm.cdf(d1))
    except Exception:
        return 0.5

def days_to_expiry(exp_str: str) -> float:
    try:
        exp = datetime.strptime(exp_str, "%Y-%m-%d")
        T = (exp - datetime.now()).days / 365.0
        return max(T, 1/365)
    except Exception:
        return 30 / 365

# ─────────────────────────────────────────────
# OPCIONES
# ─────────────────────────────────────────────

POLYGON_KEY = "PlCkWQQpdg15eQ74SEHZBdF56giLyx60"

def _make_yf_ticker(ticker: str):
    """Crea un Ticker de yfinance usando curl_cffi para evitar rate limiting."""
    try:
        from curl_cffi import requests as curl_requests
        session = curl_requests.Session(impersonate="chrome")
        return yf.Ticker(ticker, session=session)
    except Exception:
        return yf.Ticker(ticker)

def get_option_expirations(ticker: str):
    """Obtiene expiraciones via yfinance."""
    import time
    for attempt in range(3):
        try:
            tk = _make_yf_ticker(ticker)
            exps = tk.options
            today = datetime.now().strftime("%Y-%m-%d")
            return [e for e in exps if e >= today] if exps else []
        except Exception as e:
            time.sleep(2)
    return []

def load_option_chain(ticker: str, expiration: str):
    """Carga la cadena de opciones via yfinance."""
    import time
    for attempt in range(3):
        try:
            tk = _make_yf_ticker(ticker)
            chain = tk.option_chain(expiration)
            calls = chain.calls.copy()
            puts  = chain.puts.copy()
            if not calls.empty and not puts.empty:
                return calls, puts
            time.sleep(2)
        except Exception as e:
            time.sleep(2)
    return pd.DataFrame(), pd.DataFrame()

def compute_max_pain(calls: pd.DataFrame, puts: pd.DataFrame):
    if calls.empty or puts.empty: return np.nan
    all_strikes = sorted(set(calls["strike"].dropna().tolist() + puts["strike"].dropna().tolist()))
    if not all_strikes: return np.nan
    pain = []
    for s in all_strikes:
        total = 0.0
        for _, r in calls.iterrows():
            total += max(0.0, s - safe_float(r.get("strike"))) * safe_float(r.get("openInterest"), 0)
        for _, r in puts.iterrows():
            total += max(0.0, safe_float(r.get("strike")) - s) * safe_float(r.get("openInterest"), 0)
        pain.append((s, total))
    pain_df = pd.DataFrame(pain, columns=["strike", "pain"])
    return safe_float(pain_df.loc[pain_df["pain"].idxmin(), "strike"])

def compute_expected_move(calls: pd.DataFrame, puts: pd.DataFrame, spot: float):
    if calls.empty or puts.empty or pd.isna(spot):
        return np.nan, np.nan, np.nan
    strikes = sorted(set(calls["strike"].dropna().tolist()) & set(puts["strike"].dropna().tolist()))
    if not strikes: return np.nan, np.nan, np.nan
    atm = nearest_value(strikes, spot)
    c_row = calls.loc[calls["strike"] == atm]
    p_row = puts.loc[puts["strike"] == atm]
    if c_row.empty or p_row.empty: return atm, np.nan, np.nan
    c_mid = safe_float((c_row["bid"].fillna(0) + c_row["ask"].fillna(0)).iloc[0] / 2.0)
    p_mid = safe_float((p_row["bid"].fillna(0) + p_row["ask"].fillna(0)).iloc[0] / 2.0)
    if c_mid == 0 and p_mid == 0:
        em = safe_float(c_row["lastPrice"].iloc[0]) + safe_float(p_row["lastPrice"].iloc[0])
    else:
        em = c_mid + p_mid
    return atm, em, (em / spot * 100) if spot else np.nan

def compute_gex(calls: pd.DataFrame, puts: pd.DataFrame,
                spot: float, expiration: str) -> tuple:
    """
    Gamma Exposure (GEX) neto.
    GEX = sum(call_OI * gamma * spot^2 * 100) - sum(put_OI * gamma * spot^2 * 100)
    Positivo = dealers long gamma (mercado se ancla)
    Negativo = dealers short gamma (movimientos amplificados)
    Retorna (gex_total, gamma_flip_level, dealer_position, gex_by_strike_df)
    """
    if calls.empty or puts.empty or pd.isna(spot):
        return np.nan, np.nan, "N/A", pd.DataFrame()

    T = days_to_expiry(expiration)
    r = 0.05
    rows = []

    for _, row in calls.iterrows():
        K  = safe_float(row.get("strike"))
        oi = safe_float(row.get("openInterest"), 0)
        iv = safe_float(row.get("impliedVolatility"), 0.3)
        if pd.isna(K) or K <= 0 or iv <= 0: continue
        g  = bs_gamma(spot, K, T, r, iv)
        rows.append({"strike": K, "gex": g * oi * spot**2 * 0.01})

    for _, row in puts.iterrows():
        K  = safe_float(row.get("strike"))
        oi = safe_float(row.get("openInterest"), 0)
        iv = safe_float(row.get("impliedVolatility"), 0.3)
        if pd.isna(K) or K <= 0 or iv <= 0: continue
        g  = bs_gamma(spot, K, T, r, iv)
        rows.append({"strike": K, "gex": -g * oi * spot**2 * 0.01})  # puts = negativo

    if not rows:
        return np.nan, np.nan, "N/A", pd.DataFrame()

    gex_df = pd.DataFrame(rows)
    gex_by_strike = gex_df.groupby("strike")["gex"].sum().reset_index()

    total_gex = float(gex_by_strike["gex"].sum())

    # Gamma Flip: acumular GEX ordenando strikes por distancia al spot (más cercano primero)
    # El cruce de signo del cumGEX indica el nivel donde los dealers cambian de posición
    gex_by_dist = gex_by_strike.copy()
    gex_by_dist["dist"] = (gex_by_dist["strike"] - spot).abs()
    gex_by_dist = gex_by_dist.sort_values("dist").reset_index(drop=True)
    gex_by_dist["cumgex"] = gex_by_dist["gex"].cumsum()

    gamma_flip = np.nan
    for i in range(1, len(gex_by_dist)):
        prev = gex_by_dist.iloc[i - 1]["cumgex"]
        curr = gex_by_dist.iloc[i]["cumgex"]
        if pd.notna(prev) and pd.notna(curr) and prev * curr < 0:
            gamma_flip = float(gex_by_dist.iloc[i]["strike"])
            break

    if pd.isna(gamma_flip):
        # Fallback: strike con GEX individual más cercano a cero
        gex_by_strike["abs_gex"] = gex_by_strike["gex"].abs()
        gamma_flip = safe_float(gex_by_strike.loc[gex_by_strike["abs_gex"].idxmin(), "strike"])

    dealer_position = (
        "Long Gamma 🟢"  if pd.notna(total_gex) and total_gex > 0 else
        "Short Gamma 🔴" if pd.notna(total_gex) and total_gex < 0 else
        "Neutral ⚪"
    )

    # Devolver ordenado por strike para los charts
    gex_by_strike_final = gex_by_strike.drop(columns=["abs_gex"], errors="ignore").sort_values("strike")
    return total_gex, gamma_flip, dealer_position, gex_by_strike_final

def compute_vol_skew(calls: pd.DataFrame, puts: pd.DataFrame, spot: float) -> float:
    """
    Volatility skew simplificado: IV del put 10% OTM vs call 10% OTM.
    Positivo = put skew (mercado bearish/hedging)
    Negativo = call skew (mercado bullish)
    """
    if calls.empty or puts.empty or pd.isna(spot):
        return np.nan
    try:
        otm_put_strike  = spot * 0.90
        otm_call_strike = spot * 1.10
        put_row  = puts.iloc[(puts["strike"] - otm_put_strike).abs().argsort()[:1]]
        call_row = calls.iloc[(calls["strike"] - otm_call_strike).abs().argsort()[:1]]
        if put_row.empty or call_row.empty: return np.nan
        put_iv  = safe_float(put_row["impliedVolatility"].iloc[0])
        call_iv = safe_float(call_row["impliedVolatility"].iloc[0])
        if pd.isna(put_iv) or pd.isna(call_iv): return np.nan
        return put_iv - call_iv
    except Exception:
        return np.nan

def summarize_options(ticker: str, spot: float) -> OptionsSummary:
    empty = OptionsSummary(
        "N/A", np.nan, np.nan, 0, 0, np.nan, np.nan, np.nan, np.nan, np.nan,
        np.nan, np.nan, "N/A", np.nan,
        pd.DataFrame(), pd.DataFrame(),
        False, "No listed options or unavailable."
    )
    exps = get_option_expirations(ticker)
    if not exps: return empty

    expiration = exps[0]
    calls, puts = load_option_chain(ticker, expiration)
    if calls.empty or puts.empty:
        empty.expiration = expiration
        return empty

    for df in [calls, puts]:
        df["openInterest"]     = pd.to_numeric(df["openInterest"],     errors="coerce").fillna(0)
        df["impliedVolatility"] = pd.to_numeric(df["impliedVolatility"], errors="coerce").fillna(0.3)

    call_oi = int(calls["openInterest"].sum())
    put_oi  = int(puts["openInterest"].sum())
    pc_ratio = safe_float(put_oi / call_oi) if call_oi > 0 else np.nan
    max_pain = compute_max_pain(calls, puts)
    atm, em, em_pct = compute_expected_move(calls, puts, spot)
    call_wall = safe_float(calls.loc[calls["openInterest"].idxmax(), "strike"]) if not calls.empty else np.nan
    put_wall  = safe_float(puts.loc[puts["openInterest"].idxmax(),  "strike"]) if not puts.empty  else np.nan

    gex_total, gamma_flip, dealer_pos, _ = compute_gex(calls, puts, spot, expiration)
    vol_skew = compute_vol_skew(calls, puts, spot)

    return OptionsSummary(
        expiration, max_pain, pc_ratio, call_oi, put_oi,
        em, em_pct, atm, call_wall, put_wall,
        gex_total, gamma_flip, dealer_pos, vol_skew,
        calls, puts,
        True, "Full options analysis via yfinance."
    )

# ─────────────────────────────────────────────
# SENTIMENT
# ─────────────────────────────────────────────

def compute_sentiment_proxy(df: pd.DataFrame, tech: TechnicalSummary):
    last_5  = df["Close"].pct_change(5).iloc[-1]  if len(df) > 6  else np.nan
    last_20 = df["Close"].pct_change(20).iloc[-1] if len(df) > 21 else np.nan
    score = 0.5
    if pd.notna(last_5):  score += np.clip(last_5,        -0.1, 0.1)
    if pd.notna(last_20): score += np.clip(last_20 / 2,   -0.1, 0.1)
    if pd.notna(tech.rsi14):
        score += 0.05 if tech.rsi14 > 60 else (-0.05 if tech.rsi14 < 40 else 0)
    score = float(np.clip(score, 0.0, 1.0))
    label = "Positive" if score > 0.58 else "Negative" if score < 0.42 else "Neutral"
    return score, label

# ─────────────────────────────────────────────
# MONTE CARLO
# ─────────────────────────────────────────────

def monte_carlo_paths(df: pd.DataFrame, days=DEFAULT_MC_DAYS, n_paths=DEFAULT_MC_PATHS):
    returns = df["Close"].pct_change().dropna()
    if len(returns) < 30: return None
    mu, sigma  = returns.mean(), returns.std()
    last_price = safe_float(df["Close"].iloc[-1])
    if pd.isna(last_price): return None
    rng = np.random.default_rng()
    shocks = rng.normal(mu, sigma, (n_paths, days))
    paths  = np.cumprod(1 + shocks, axis=1)
    sim    = last_price * np.hstack([np.ones((n_paths, 1)), paths])
    sim_df = pd.DataFrame(sim.T)
    sim_df.index = range(0, days + 1)
    return sim_df

def summarize_monte_carlo(sim_df: pd.DataFrame, last_price: float) -> MonteCarloSummary:
    final = sim_df.iloc[-1]
    return MonteCarloSummary(
        expected_close = safe_float(final.mean()),
        expected_high  = safe_float(sim_df.max().max()),
        expected_low   = safe_float(sim_df.min().min()),
        p10_close      = safe_float(final.quantile(0.10)),
        p50_close      = safe_float(final.quantile(0.50)),
        p90_close      = safe_float(final.quantile(0.90)),
        bullish_prob   = float((final > last_price * 1.01).mean()),
        bearish_prob   = float((final < last_price * 0.99).mean()),
        neutral_prob   = float(((final >= last_price * 0.99) & (final <= last_price * 1.01)).mean()),
    )

# ─────────────────────────────────────────────
# ELITE DECISION ENGINE
# ─────────────────────────────────────────────

def elite_decision_engine(ticker, df, tech, opt, mc, sentiment_score, sentiment_label) -> EliteSummary:
    score = 0.5
    score += 0.12 if tech.trend_signal    == "Bullish" else (-0.12 if tech.trend_signal    == "Bearish" else 0)
    score += 0.08 if tech.momentum_signal == "Bullish" else (-0.08 if tech.momentum_signal == "Bearish" else 0)
    if pd.notna(tech.rsi14):
        score += 0.05 if tech.rsi14 > 60 else (-0.05 if tech.rsi14 < 40 else 0)
    if pd.notna(tech.macd_hist):
        score += 0.04 if tech.macd_hist > 0 else -0.04
    score += (sentiment_score - 0.5) * 0.20

    notes = []
    options_mode = "Unavailable"
    if opt.options_available:
        options_mode = "Enabled"
        if pd.notna(opt.max_pain):
            score += 0.01 if tech.price > opt.max_pain else -0.01
        if pd.notna(opt.put_call_oi_ratio):
            if opt.put_call_oi_ratio > 1.2:
                notes.append("Heavy put positioning.")
                score -= 0.04
            elif opt.put_call_oi_ratio < 0.8:
                notes.append("Call-heavy positioning.")
                score += 0.04
        if pd.notna(opt.call_wall) and pd.notna(opt.put_wall):
            if tech.price >= opt.call_wall * 0.99:
                notes.append("Near call wall / resistance.")
                score -= 0.03
            if tech.price <= opt.put_wall * 1.01:
                notes.append("Near put wall / support.")
                score += 0.03
        # GEX influence
        if pd.notna(opt.gex):
            if opt.gex > 0:
                notes.append("Dealers long gamma — expect mean reversion.")
            elif opt.gex < 0:
                notes.append("Dealers short gamma — expect amplified moves.")
        if pd.notna(opt.gamma_flip):
            if tech.price > opt.gamma_flip:
                notes.append(f"Above gamma flip ({opt.gamma_flip:.2f}) — bullish regime.")
                score += 0.03
            else:
                notes.append(f"Below gamma flip ({opt.gamma_flip:.2f}) — bearish regime.")
                score -= 0.03

    score += (mc.bullish_prob - mc.bearish_prob) * 0.20
    score  = float(np.clip(score, 0.0, 1.0))

    direction = "Bullish" if score > 0.57 else ("Bearish" if score < 0.43 else "Neutral")
    strategy  = {
        "Bullish": "Favor pullback buys / trend continuation.",
        "Bearish": "Favor rallies to fade / defensive posture.",
        "Neutral": "Range / wait for confirmation.",
    }[direction]

    intraday_range = tech.atr14 if pd.notna(tech.atr14) else tech.price * 0.02
    if pd.notna(opt.expected_move):
        intraday_range = min(max(intraday_range, opt.expected_move * 0.35), opt.expected_move)

    notes.append(f"Sentiment: {sentiment_label}.")
    if pd.notna(opt.expected_move):
        notes.append(f"Expected move: {opt.expected_move:.2f} ({opt.implied_move_pct:.2f}%).")
    if pd.notna(opt.vol_skew_25d):
        skew_dir = "put skew" if opt.vol_skew_25d > 0 else "call skew"
        notes.append(f"Vol skew: {opt.vol_skew_25d:.3f} ({skew_dir}).")

    return EliteSummary(
        ticker, direction, classify_confidence(score), tech.price,
        safe_float(tech.price + intraday_range / 2),
        safe_float(tech.price - intraday_range / 2),
        safe_float(mc.expected_close),
        safe_float(mc.expected_high),
        safe_float(mc.expected_low),
        safe_float(mc.expected_close),
        strategy, options_mode, " ".join(notes)
    )

# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────

def run_nasa_elite(ticker, period, interval, mc_days, mc_paths):
    raw = download_market_data(ticker, period=period, interval=interval)
    if raw.empty or len(raw) < 60:
        return {"ok": False, "ticker": ticker, "error": "Insufficient market data."}
    df   = add_indicators(raw)
    tech = summarize_technicals(df)
    opt  = summarize_options(ticker, tech.price)
    sentiment_score, sentiment_label = compute_sentiment_proxy(df, tech)
    sim_df = monte_carlo_paths(df, days=mc_days, n_paths=mc_paths)
    if sim_df is None:
        return {"ok": False, "ticker": ticker, "error": "Monte Carlo failed."}
    mc    = summarize_monte_carlo(sim_df, tech.price)
    elite = elite_decision_engine(ticker, df, tech, opt, mc, sentiment_score, sentiment_label)
    return {"ok": True, "ticker": ticker, "raw": raw, "df": df,
            "technicals": tech, "options": opt,
            "sentiment_score": sentiment_score, "sentiment_label": sentiment_label,
            "sim_df": sim_df, "mc": mc, "elite": elite}

# ─────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────

def fmt(x):
    if pd.isna(x): return "N/A"
    return f"{x:,.2f}"

def metric4(v1, v2, v3, v4, labels):
    cols = st.columns(4)
    for col, v, l in zip(cols, [v1, v2, v3, v4], labels):
        col.metric(l, v)

# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────

st.title("🚀 NASA v4.0 ELITE")
st.caption("Multi-ticker · Técnicos · GEX · Dealer Positioning · Gamma Flip · Vol Skew · OI Heatmap · Monte Carlo 10K")

with st.sidebar:
    st.header("⚙️ Configuración")
    tickers_input = st.text_input("Tickers (separados por coma)", value="QQQ,TQQQ,NVDA,MSFT")
    period        = st.selectbox("Período histórico", ["6mo", "1y", "2y", "5y"], index=1)
    interval      = st.selectbox("Intervalo", ["1d", "1h"], index=0)
    mc_days       = st.number_input("Horizonte Monte Carlo (días)", 1, 30, 5, 1)
    mc_paths      = st.number_input("Paths Monte Carlo", 1000, 25000, 10000, 1000)
    run_btn       = st.button("▶ Ejecutar análisis", type="primary", use_container_width=True)

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
if not tickers:
    st.warning("Ingresá al menos un ticker.")
    st.stop()

if not run_btn:
    st.info("Configurá los tickers en el panel izquierdo y presioná **▶ Ejecutar análisis**.")
    st.stop()

all_summary_rows = []

for ticker in tickers:
    st.markdown("---")
    st.header(f"📊 {ticker}")

    with st.spinner(f"Analizando {ticker}..."):
        result = run_nasa_elite(ticker, period, interval, int(mc_days), int(mc_paths))

    if not result["ok"]:
        st.error(f"{ticker}: {result['error']}")
        continue

    df    = result["df"]
    tech  = result["technicals"]
    opt   = result["options"]
    mc    = result["mc"]
    elite = result["elite"]

    # ── KPIs principales ──
    metric4(
        fmt(elite.last_price), elite.direction,
        f"{elite.confidence:.0f}%", opt.dealer_position,
        ["Precio", "Dirección", "Confianza", "Dealer Position"]
    )

    tab_elite, tab_gex, tab_tech, tab_opt, tab_mc, tab_data = st.tabs([
        "🏆 Resumen ELITE", "⚡ GEX & Gamma", "📈 Técnicos",
        "🎯 Opciones", "🎲 Monte Carlo", "📋 Datos"
    ])

    # ── TAB 1: RESUMEN ELITE ──
    with tab_elite:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Estimaciones de precio")
            est_df = pd.DataFrame([{
                "Intradía Alto Est.":  fmt(elite.intraday_high_est),
                "Intradía Bajo Est.":  fmt(elite.intraday_low_est),
                "Cierre Diario Est.":  fmt(elite.daily_close_est),
                "Semanal Alto Est.":   fmt(elite.weekly_high_est),
                "Semanal Bajo Est.":   fmt(elite.weekly_low_est),
                "Cierre Semanal Est.": fmt(elite.weekly_close_est),
            }])
            st.dataframe(est_df, use_container_width=True)
            st.info(f"**Estrategia:** {elite.strategy}")
            st.caption(elite.notes)

        with c2:
            st.subheader("Niveles clave")
            kl_df = pd.DataFrame([{
                "Soporte":      fmt(tech.support),
                "Resistencia":  fmt(tech.resistance),
                "BB Superior":  fmt(tech.bb_upper),
                "BB Media":     fmt(tech.bb_mid),
                "BB Inferior":  fmt(tech.bb_lower),
                "ATR14":        fmt(tech.atr14),
                "Max Pain":     fmt(opt.max_pain),
                "Call Wall":    fmt(opt.call_wall),
                "Put Wall":     fmt(opt.put_wall),
                "Gamma Flip":   fmt(opt.gamma_flip),
            }])
            st.dataframe(kl_df, use_container_width=True)

    # ── TAB 2: GEX & GAMMA ──
    with tab_gex:
        if opt.options_available:
            g1, g2, g3, g4 = st.columns(4)
            g1.metric("GEX Total", fmt(opt.gex))
            g2.metric("Gamma Flip", fmt(opt.gamma_flip))
            g3.metric("Dealer Position", opt.dealer_position)
            g4.metric("Vol Skew (10d OTM)", f"{opt.vol_skew_25d:.3f}" if pd.notna(opt.vol_skew_25d) else "N/A")

            st.markdown("---")

            # GEX por strike (heatmap de barras)
            if not opt.calls_df.empty and not opt.puts_df.empty:
                st.subheader("📊 GEX por Strike")
                calls_c = opt.calls_df.copy()
                puts_c  = opt.puts_df.copy()
                calls_c["openInterest"] = pd.to_numeric(calls_c["openInterest"], errors="coerce").fillna(0)
                puts_c["openInterest"]  = pd.to_numeric(puts_c["openInterest"],  errors="coerce").fillna(0)
                calls_c["impliedVolatility"] = pd.to_numeric(calls_c["impliedVolatility"], errors="coerce").fillna(0.3)
                puts_c["impliedVolatility"]  = pd.to_numeric(puts_c["impliedVolatility"],  errors="coerce").fillna(0.3)

                T = days_to_expiry(opt.expiration)
                rows_gex = []
                for _, r in calls_c.iterrows():
                    K = safe_float(r.get("strike"))
                    oi = safe_float(r.get("openInterest"), 0)
                    iv = safe_float(r.get("impliedVolatility"), 0.3)
                    if K > 0 and iv > 0:
                        g = bs_gamma(tech.price, K, T, 0.05, iv)
                        rows_gex.append({"strike": K, "GEX": g * oi * tech.price**2 * 0.01})
                for _, r in puts_c.iterrows():
                    K = safe_float(r.get("strike"))
                    oi = safe_float(r.get("openInterest"), 0)
                    iv = safe_float(r.get("impliedVolatility"), 0.3)
                    if K > 0 and iv > 0:
                        g = bs_gamma(tech.price, K, T, 0.05, iv)
                        rows_gex.append({"strike": K, "GEX": -g * oi * tech.price**2 * 0.01})

                if rows_gex:
                    gex_plot = pd.DataFrame(rows_gex).groupby("strike")["GEX"].sum().reset_index()
                    gex_plot = gex_plot.sort_values("strike")
                    # Filtrar rango cercano al precio (±20%)
                    gex_plot = gex_plot[
                        (gex_plot["strike"] >= tech.price * 0.80) &
                        (gex_plot["strike"] <= tech.price * 1.20)
                    ]
                    gex_plot = gex_plot.set_index("strike")
                    st.bar_chart(gex_plot)

                # OI Heatmap por strike
                st.subheader("📊 Open Interest por Strike")
                oi_calls = calls_c[["strike", "openInterest"]].copy()
                oi_calls.columns = ["strike", "Call OI"]
                oi_puts  = puts_c[["strike", "openInterest"]].copy()
                oi_puts.columns = ["strike", "Put OI"]
                oi_merged = pd.merge(oi_calls, oi_puts, on="strike", how="outer").fillna(0)
                oi_merged = oi_merged[
                    (oi_merged["strike"] >= tech.price * 0.80) &
                    (oi_merged["strike"] <= tech.price * 1.20)
                ].sort_values("strike").set_index("strike")
                st.bar_chart(oi_merged)

                # Volatility Skew
                st.subheader("📉 Volatility Skew por Strike")
                iv_calls = calls_c[["strike", "impliedVolatility"]].copy()
                iv_calls.columns = ["strike", "Call IV"]
                iv_puts  = puts_c[["strike", "impliedVolatility"]].copy()
                iv_puts.columns = ["strike", "Put IV"]
                iv_merged = pd.merge(iv_calls, iv_puts, on="strike", how="outer").fillna(np.nan)
                iv_merged = iv_merged[
                    (iv_merged["strike"] >= tech.price * 0.80) &
                    (iv_merged["strike"] <= tech.price * 1.20)
                ].sort_values("strike").set_index("strike")
                st.line_chart(iv_merged)

        else:
            st.warning("Opciones no disponibles para este ticker.")

    # ── TAB 3: TÉCNICOS ──
    with tab_tech:
        tech_df = pd.DataFrame([{
            "Precio": fmt(tech.price), "SMA20": fmt(tech.sma20),
            "SMA50": fmt(tech.sma50), "SMA200": fmt(tech.sma200),
            "RSI14": fmt(tech.rsi14), "MACD": fmt(tech.macd),
            "MACD Hist": fmt(tech.macd_hist), "ATR14": fmt(tech.atr14),
            "Vol 20d Ann.": fmt(tech.volatility20),
            "Trend": tech.trend_signal, "Momentum": tech.momentum_signal,
        }])
        st.dataframe(tech_df, use_container_width=True)
        st.line_chart(df[["Close", "SMA20", "SMA50", "SMA200"]])
        st.line_chart(df[["Close", "BB_UPPER", "BB_MID", "BB_LOWER"]])
        st.line_chart(df[["MACD", "MACD_SIGNAL", "MACD_HIST"]])
        st.line_chart(df[["RSI14"]])

    # ── TAB 4: OPCIONES ──
    with tab_opt:
        if opt.options_available:
            o1, o2, o3, o4 = st.columns(4)
            o1.metric("Max Pain",       fmt(opt.max_pain))
            o2.metric("Expected Move",  fmt(opt.expected_move))
            o3.metric("Put/Call OI",    fmt(opt.put_call_oi_ratio))
            o4.metric("Implied Move %", f"{fmt(opt.implied_move_pct)}%")

            o5, o6, o7, o8 = st.columns(4)
            o5.metric("Call Wall",  fmt(opt.call_wall))
            o6.metric("Put Wall",   fmt(opt.put_wall))
            o7.metric("ATM Strike", fmt(opt.atm_strike))
            o8.metric("Expiration", opt.expiration)

            if pd.notna(opt.expected_move):
                em_up   = tech.price + opt.expected_move
                em_down = tech.price - opt.expected_move
                st.success(f"Rango esperado: **{fmt(em_down)} – {fmt(em_up)}**")

            st.subheader("Chain de Calls")
            st.dataframe(opt.calls_df[["strike","lastPrice","bid","ask","openInterest","impliedVolatility","volume"]].head(20), use_container_width=True)
            st.subheader("Chain de Puts")
            st.dataframe(opt.puts_df[["strike","lastPrice","bid","ask","openInterest","impliedVolatility","volume"]].head(20), use_container_width=True)
        else:
            st.warning("Opciones no disponibles.")

    # ── TAB 5: MONTE CARLO ──
    with tab_mc:
        m1, m2, m3 = st.columns(3)
        m1.metric("Expected Close", fmt(mc.expected_close))
        m2.metric("P10 Close",      fmt(mc.p10_close))
        m3.metric("P90 Close",      fmt(mc.p90_close))

        m4, m5, m6 = st.columns(3)
        m4.metric("Bullish Prob", f"{mc.bullish_prob:.1%}")
        m5.metric("Neutral Prob", f"{mc.neutral_prob:.1%}")
        m6.metric("Bearish Prob", f"{mc.bearish_prob:.1%}")

        st.subheader(f"Monte Carlo — {int(mc_paths):,} trayectorias / {int(mc_days)} días")
        st.line_chart(result["sim_df"].iloc[:, :min(300, result["sim_df"].shape[1])])

    # ── TAB 6: DATOS ──
    with tab_data:
        st.dataframe(df.tail(50), use_container_width=True)

    all_summary_rows.append({
        "Ticker": elite.ticker, "Direction": elite.direction,
        "Confidence %": elite.confidence, "Price": elite.last_price,
        "Intraday High": elite.intraday_high_est, "Intraday Low": elite.intraday_low_est,
        "Daily Close Est.": elite.daily_close_est,
        "GEX": opt.gex, "Gamma Flip": opt.gamma_flip,
        "Dealer Position": opt.dealer_position,
        "Vol Skew": opt.vol_skew_25d, "Max Pain": opt.max_pain,
        "Expected Move": opt.expected_move, "Strategy": elite.strategy,
    })

# ── RESUMEN GLOBAL ──
if all_summary_rows:
    st.markdown("---")
    st.header("🌐 Resumen Global NASA v4.0 ELITE")
    global_df = pd.DataFrame(all_summary_rows)
    st.dataframe(global_df, use_container_width=True)
    csv = global_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Descargar resumen CSV", data=csv,
        file_name=f"nasa_v4_elite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    st.success("✅ NASA v4.0 ELITE — análisis completado.")
