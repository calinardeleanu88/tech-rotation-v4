import os, json, time
import numpy as np
import pandas as pd
import yfinance as yf

# =========================
# CONFIG
# =========================

# Buckets
PLATFORMS = ["MSFT", "GOOGL", "META"]
AI_SEMIS  = ["NVDA", "AMD"]
EQUIP     = ["KLAC"]
MEMORY    = ["MU", "WDC", "STX", "SNDK"]   # basket B
INFRA_OPT = ["VRT"]                        # optional (may be bad on GH)

DEF = ["XLV", "IEF"]

# Macro proxies
QQQ = "QQQ"
SPY = "SPY"
SMH = "SMH"  # for RS comparison (memory vs semis proxy)

START = "2006-01-01"
END = None

# Macro gates
QQQ_EMA_W = 30
SPY_EMA_W = 40

# Internal signals
WEEKLY_GATE_EMA_W = 30
DAILY_SMA = 20
CONF_WIN  = 10
CONF_ON   = 7
CONF_OFF  = 3
RS_WIN_D  = 63  # ~3 months

# Execution frictions
MIN_CHANGE_TO_TRADE = 0.05
COOLDOWN_DAYS = 15
TC_BPS_PER_1X = 5

# DEF split
DEF_XLV = 0.50
DEF_IEF = 0.50

# Target weights at RiskBudget=1.0 (aggressive, but controlled)
W_PLAT = 0.30
W_AI   = 0.25
W_EQ   = 0.15
W_INF  = 0.10
W_MEM  = 0.20

# Memory guard parameters (B)
MEM_TREND_EMA_W = 40
MEM_RS_WIN_D = 63          # 12 weeks-ish
MEM_ROC_D = 63             # 3 months ROC
MEM_SHOCK_LOOKBACK_W = 8   # 8 weeks
MEM_SHOCK_DD = -0.20       # -20%
MEM_NEG_WEEKS = 4          # 4 consecutive red weeks

OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

# =========================
# Helpers
# =========================

def download_prices(tickers, start, end, tries=3, sleep_s=2):
    last_err = None
    for _ in range(tries):
        try:
            df = yf.download(
                tickers, start=start, end=end,
                auto_adjust=True, progress=False, threads=False
            )
            if df is None or len(df) == 0:
                raise RuntimeError("yfinance returned empty dataframe")
            if isinstance(df.columns, pd.MultiIndex):
                px = df["Close"]
            else:
                px = df.get("Close", df)
            if isinstance(px, pd.Series):
                px = px.to_frame()
            return px
        except Exception as e:
            last_err = e
            time.sleep(sleep_s)
    raise RuntimeError(f"Failed to download prices after {tries} tries: {last_err}")

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def normalize_positive(s: pd.Series) -> pd.Series:
    s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    tot = float(s.sum())
    return s*0.0 if tot <= 0 else s/tot

def add_def(row: pd.Series, amount: float):
    if amount <= 0: return
    row["XLV"] += amount * DEF_XLV
    row["IEF"] += amount * DEF_IEF

def max_drawdown(eq: pd.Series) -> float:
    peak = eq.cummax()
    return float((eq/peak - 1.0).min())

def cagr(eq: pd.Series) -> float:
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    return float((eq.iloc[-1]/eq.iloc[0])**(1/years) - 1) if years > 0 else np.nan

def ann_vol(r: pd.Series) -> float:
    return float(r.std() * np.sqrt(252))

def sharpe(r: pd.Series) -> float:
    v = r.std()
    return float((r.mean()/v) * np.sqrt(252)) if v != 0 else np.nan

def sortino(r: pd.Series) -> float:
    d = r[r < 0].std()
    return float((r.mean()/d) * np.sqrt(252)) if d != 0 else np.nan

def trading_days_between(idx: pd.DatetimeIndex, a, b) -> int:
    if a is None: return 10**9
    return int(((idx > a) & (idx <= b)).sum())

def equal_weight_index(px: pd.DataFrame, tickers: list[str]) -> pd.Series:
    cols = [t for t in tickers if t in px.columns]
    if not cols:
        return pd.Series(np.nan, index=px.index)
    return px[cols].mean(axis=1)

# =========================
# Data
# =========================

ALL = sorted(set(
    PLATFORMS + AI_SEMIS + EQUIP + MEMORY + INFRA_OPT + DEF + [QQQ, SPY, SMH]
))

px = download_prices(ALL, START, END).dropna(how="all").ffill()
if px.empty:
    raise RuntimeError("Price data empty.")

# Diagnostics coverage
coverage = px.notna().mean()
diag = pd.DataFrame({
    "ticker": coverage.index,
    "coverage": coverage.values,
    "last_price": [px[c].dropna().iloc[-1] if px[c].notna().any() else np.nan for c in px.columns],
    "last_date":  [str(px[c].dropna().index[-1].date()) if px[c].notna().any() else "" for c in px.columns],
})
diag.to_csv(os.path.join(OUTDIR, "diagnostics.csv"), index=False)

# Hard fail if core missing badly (not infra optional)
CORE = set(PLATFORMS + AI_SEMIS + EQUIP + MEMORY + DEF + [QQQ, SPY, SMH])
bad_core = [t for t in coverage.index if (t in CORE and float(coverage[t]) < 0.80)]
if bad_core:
    raise RuntimeError(f"Data coverage too low for CORE tickers: {bad_core} — see outputs/diagnostics.csv")

# Infra optional
infra_enabled = True
for t in INFRA_OPT:
    if t in coverage.index and float(coverage[t]) < 0.80:
        infra_enabled = False

# Returns / weekly
ret_d = px.pct_change().fillna(0.0)
px_w = px.resample("W-FRI").last().dropna(how="all").ffill()
ret_w = px_w.pct_change().fillna(0.0)

# =========================
# Macro Risk Budget
# =========================

qqq_w = px_w[QQQ]
spy_w = px_w[SPY]

qqq_gate_w = (qqq_w > ema(qqq_w, QQQ_EMA_W)).astype(bool)
spy_gate_w = (spy_w > ema(spy_w, SPY_EMA_W)).astype(bool)

# Align to daily via ffill
qqq_gate = qqq_gate_w.reindex(px.index, method="ffill").fillna(False)
spy_gate = spy_gate_w.reindex(px.index, method="ffill").fillna(False)

# RiskBudget: 1.0 / 0.5 / 0.0
risk_budget = pd.Series(0.0, index=px.index)
risk_budget[spy_gate & (~qqq_gate)] = 0.5
risk_budget[qqq_gate] = 1.0  # if qqq_gate True => 1.0 regardless of spy_gate

# =========================
# Internal stock eligibility (daily confirm + weekly gate)
# =========================

# Weekly gate for individual stocks (trend filter)
weekly_gate = {}
for t in (PLATFORMS + AI_SEMIS + EQUIP + MEMORY + (INFRA_OPT if infra_enabled else [])):
    w = px_w[t]
    weekly_gate[t] = (w > ema(w, WEEKLY_GATE_EMA_W)).astype(bool).reindex(px.index, method="ffill").fillna(False)
weekly_gate = pd.DataFrame(weekly_gate, index=px.index)

# Daily confirm: SMA20 and 7/10 ON, 3/10 OFF
sma = px.rolling(DAILY_SMA).mean()
above = (px > sma).astype(bool).fillna(False)
cnt = above.rolling(CONF_WIN).sum()

confirm_on  = (cnt >= CONF_ON).fillna(False)
confirm_off = (cnt <= CONF_OFF).fillna(False)

def build_state(ticker: str) -> pd.Series:
    gate = weekly_gate[ticker].astype(bool)
    on_sig  = confirm_on[ticker].astype(bool) & gate
    off_sig = confirm_off[ticker].astype(bool) | (~gate)

    st = pd.Series(False, index=px.index, dtype=bool)
    on = False
    for dt in px.index:
        if bool(off_sig.loc[dt]): on = False
        elif bool(on_sig.loc[dt]): on = True
        st.loc[dt] = on
    return st

STATE_UNI = (PLATFORMS + AI_SEMIS + EQUIP + MEMORY + (INFRA_OPT if infra_enabled else []))
state = pd.DataFrame({t: build_state(t) for t in STATE_UNI}, index=px.index)

# Relative strength measure (3M)
rs = px.pct_change(RS_WIN_D).replace([np.inf, -np.inf], np.nan).fillna(0.0)

# =========================
# Memory Guard B
# =========================

mem_idx = equal_weight_index(px, MEMORY)
mem_w = mem_idx.resample("W-FRI").last().ffill()
mem_ema40w = ema(mem_w, MEM_TREND_EMA_W)

# (A) Trend hard kill
mem_trend_ok_w = (mem_w > mem_ema40w).astype(bool)
mem_trend_ok = mem_trend_ok_w.reindex(px.index, method="ffill").fillna(False)

# (B) RS vs SMH (soft cut)
ratio = (mem_idx / px[SMH]).replace([np.inf, -np.inf], np.nan).ffill()
mem_rs = ratio.pct_change(MEM_RS_WIN_D).fillna(0.0)
mem_rs_ok = (mem_rs > 0)

# (C) Momentum cap (ROC 3M)
mem_roc = mem_idx.pct_change(MEM_ROC_D).fillna(0.0)
mem_mom_ok = (mem_roc > 0)

# (D) Shock trigger
lookback = MEM_SHOCK_LOOKBACK_W
mem_roll_peak = mem_w.rolling(lookback).max()
mem_dd_8w = (mem_w / mem_roll_peak - 1.0).fillna(0.0)
shock_dd = (mem_dd_8w <= MEM_SHOCK_DD)

neg_w = mem_w.pct_change().fillna(0.0)
neg_streak = (neg_w < 0).rolling(MEM_NEG_WEEKS).sum() >= MEM_NEG_WEEKS

mem_shock_w = (shock_dd | neg_streak).astype(bool)
mem_shock = mem_shock_w.reindex(px.index, method="ffill").fillna(False)

def memory_weight_cap(dt) -> float:
    if not bool(mem_trend_ok.loc[dt]):  # hard kill
        return 0.0
    if bool(mem_shock.loc[dt]):         # panic kill
        return 0.0

    cap = W_MEM
    if not bool(mem_rs_ok.loc[dt]):
        cap = min(cap, W_MEM * 0.5)
    if not bool(mem_mom_ok.loc[dt]):
        cap = min(cap, 0.12)  # 10–12%
    return cap

# =========================
# Build target weights daily
# =========================

assets = (PLATFORMS + AI_SEMIS + EQUIP + (INFRA_OPT if infra_enabled else []) + MEMORY + DEF)
w_target = pd.DataFrame(0.0, index=px.index, columns=assets)

def alloc_bucket(dt, names, budget):
    elig = [t for t in names if (t in state.columns and bool(state.loc[dt, t]))]
    if not elig or budget <= 0:
        return None
    scores = normalize_positive(rs.loc[dt, elig])
    if float(scores.sum()) == 0:
        return {t: budget/len(elig) for t in elig}
    return {t: budget*float(scores.loc[t]) for t in elig}

for dt in px.index:
    rb = float(risk_budget.loc[dt])  # 0/0.5/1
    row = w_target.loc[dt]
    row[:] = 0.0

    if rb <= 0:
        add_def(row, 1.0)
        continue

    b_plat = W_PLAT * rb
    b_ai   = W_AI   * rb
    b_eq   = W_EQ   * rb
    b_inf  = (W_INF * rb) if infra_enabled else 0.0

    mem_cap = memory_weight_cap(dt) * rb
    b_mem = mem_cap

    unalloc = 0.0

    alloc = alloc_bucket(dt, PLATFORMS, b_plat)
    if alloc is None: unalloc += b_plat
    else:
        for k,v in alloc.items(): row[k] += v

    alloc = alloc_bucket(dt, AI_SEMIS, b_ai)
    if alloc is None: unalloc += b_ai
    else:
        for k,v in alloc.items(): row[k] += v

    alloc = alloc_bucket(dt, EQUIP, b_eq)
    if alloc is None: unalloc += b_eq
    else:
        for k,v in alloc.items(): row[k] += v

    if b_inf > 0:
        t = INFRA_OPT[0]
        if t in state.columns and bool(state.loc[dt, t]):
            row[t] += b_inf
        else:
            unalloc += b_inf

    if b_mem > 0:
        alloc = alloc_bucket(dt, MEMORY, b_mem)
        if alloc is None:
            unalloc += b_mem
        else:
            for k,v in alloc.items(): row[k] += v

    # "Memories exit first": if mem is killed, half of intended mem sleeve to platforms, half to DEF
    if rb > 0 and (mem_cap == 0.0) and (W_MEM * rb > 0):
        lost = (W_MEM * rb)
        boost = 0.5 * lost
        alloc = alloc_bucket(dt, PLATFORMS, boost)
        if alloc is None:
            unalloc += boost
        else:
            for k,v in alloc.items(): row[k] += v
        unalloc += (lost - boost)

    def_amt = (1.0 - rb) + unalloc
    add_def(row, def_amt)

    s = float(row.sum())
    if s > 0 and abs(s - 1.0) > 1e-10:
        w_target.loc[dt] = row / s

# =========================
# Execution: DAILY with cooldown + emergency override
# =========================

exec_days = px.index
w_exec = pd.DataFrame(index=exec_days, columns=assets, data=np.nan)

prev_exec = None
last_trade_day = None
churn_flags = 0
traded_today = pd.Series(False, index=exec_days)

rb_prev = risk_budget.shift(1).fillna(risk_budget.iloc[0])
mem_shock_prev = mem_shock.shift(1).fillna(False)
emergency = ((risk_budget < rb_prev) | (mem_shock & (~mem_shock_prev)))

for dt in exec_days:
    prop = w_target.loc[dt].copy()

    if prev_exec is None:
        w_exec.loc[dt] = prop
        prev_exec = prop
        last_trade_day = dt
        traded_today.loc[dt] = True
        continue

    change = float((prop - prev_exec).abs().sum())
    td = trading_days_between(px.index, last_trade_day, dt)
    is_emergency = bool(emergency.loc[dt])

    if (not is_emergency) and (td < COOLDOWN_DAYS) and (change >= MIN_CHANGE_TO_TRADE):
        w_exec.loc[dt] = prev_exec
        churn_flags += 1
        traded_today.loc[dt] = False
    else:
        if change >= MIN_CHANGE_TO_TRADE or is_emergency:
            last_trade_day = dt
            traded_today.loc[dt] = True
        else:
            traded_today.loc[dt] = False
        w_exec.loc[dt] = prop
        prev_exec = prop

# hard normalize
row_sums = w_exec.fillna(0.0).sum(axis=1)
zero_rows = row_sums == 0
if zero_rows.any():
    w_exec.loc[zero_rows, :] = 0.0
    w_exec.loc[zero_rows, "XLV"] = DEF_XLV
    w_exec.loc[zero_rows, "IEF"] = DEF_IEF
    row_sums = w_exec.fillna(0.0).sum(axis=1)

need_norm = (row_sums > 0) & (abs(row_sums - 1.0) > 1e-8)
w_exec.loc[need_norm, :] = w_exec.loc[need_norm, :].div(row_sums[need_norm], axis=0)

w_daily = w_exec.infer_objects(copy=False).ffill().fillna(0.0)

# =========================
# Performance
# =========================

tc = TC_BPS_PER_1X / 10_000.0
dw = w_daily.diff().abs().sum(axis=1).fillna(0.0)
cost = tc * dw

port_ret = (w_daily * ret_d[assets]).sum(axis=1) - cost
equity = (1 + port_ret).cumprod()

qqq_ret = ret_d[QQQ]
spy_ret = ret_d[SPY]
eq_qqq = (1 + qqq_ret).cumprod()
eq_spy = (1 + spy_ret).cumprod()

summary = {
    "period_start": str(equity.index[0].date()),
    "period_end": str(equity.index[-1].date()),
    "cagr": cagr(equity),
    "maxdd": max_drawdown(equity),
    "vol": ann_vol(port_ret),
    "sharpe": sharpe(port_ret),
    "sortino": sortino(port_ret),
    "turnover_annual": float(dw.mean() * 252 / 2),
    "avg_def_weight": float(w_daily[DEF].sum(axis=1).mean()),
    "avg_risk_weight": float(1.0 - w_daily[DEF].sum(axis=1).mean()),
    "churn_flags": int(churn_flags),
    "infra_enabled": bool(infra_enabled),
    "latest_date": str(equity.index[-1].date()),
}

latest_dt = equity.index[-1]
macro_debug = {
    "date": str(latest_dt.date()),
    "risk_budget": float(risk_budget.loc[latest_dt]),
    "qqq_gate": bool(qqq_gate.loc[latest_dt]),
    "spy_gate": bool(spy_gate.loc[latest_dt]),
    "emergency_today": bool(emergency.loc[latest_dt]),
    "traded_today": bool(traded_today.loc[latest_dt]),
}

memory_debug = {
    "date": str(latest_dt.date()),
    "mem_trend_ok": bool(mem_trend_ok.loc[latest_dt]),
    "mem_rs_ok": bool(mem_rs_ok.loc[latest_dt]),
    "mem_mom_ok": bool(mem_mom_ok.loc[latest_dt]),
    "mem_shock": bool(mem_shock.loc[latest_dt]),
    "mem_weight_cap_base": float(memory_weight_cap(latest_dt)),
    "mem_rs_value": float(mem_rs.loc[latest_dt]),
    "mem_roc_3m": float(mem_roc.loc[latest_dt]),
}

with open(os.path.join(OUTDIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

with open(os.path.join(OUTDIR, "macro_debug.json"), "w") as f:
    json.dump(macro_debug, f, indent=2)

with open(os.path.join(OUTDIR, "memory_debug.json"), "w") as f:
    json.dump(memory_debug, f, indent=2)

w_out = w_exec.copy()
w_out["WEIGHT_SUM"] = w_exec.fillna(0.0).sum(axis=1)
w_out.tail(15).to_csv(os.path.join(OUTDIR, "recent_exec_weights.csv"), float_format="%.6f")

latest_weights = w_exec.iloc[-1].fillna(0.0).to_dict()
with open(os.path.join(OUTDIR, "latest_exec_weights.json"), "w") as f:
    json.dump(latest_weights, f, indent=2)

pd.DataFrame({"equity": equity, "port_ret": port_ret}).to_csv(os.path.join(OUTDIR, "equity.csv"))
pd.DataFrame({"equity": eq_qqq, "ret": qqq_ret}).to_csv(os.path.join(OUTDIR, "equity_qqq.csv"))
pd.DataFrame({"equity": eq_spy, "ret": spy_ret}).to_csv(os.path.join(OUTDIR, "equity_spy.csv"))

# =========================
# Run Monitor output (macro-monitor style)
# =========================

def fmt_pct(x):
    return f"{x*100:.2f}%"

def line():
    print("="*72)

line()
print("===== TECH ROTATION v4 (Macro + Internal + Memory Guard B) =====")
print(f"Period: {summary['period_start']} → {summary['period_end']}")
print(f"Costs: {TC_BPS_PER_1X} bps per 1.0 abs weight change | Cooldown: {COOLDOWN_DAYS}d | Min-change: {int(MIN_CHANGE_TO_TRADE*100)}%")
print(f"Infra enabled: {infra_enabled} | Universe: Platforms={PLATFORMS}, AI={AI_SEMIS}, Equip={EQUIP}, Mem={MEMORY}, DEF={DEF}")
line()

print("===== MACRO REGIME =====")
print(f"Date: {macro_debug['date']}")
print(f"QQQ gate (EMA{QQQ_EMA_W}W): {'ON' if macro_debug['qqq_gate'] else 'OFF'}")
print(f"SPY gate (EMA{SPY_EMA_W}W): {'ON' if macro_debug['spy_gate'] else 'OFF'}")
print(f"RISK BUDGET: {macro_debug['risk_budget']:.1f}  | Emergency today: {macro_debug['emergency_today']} | Traded today: {macro_debug['traded_today']}")
line()

print("===== MEMORY GUARD =====")
print(f"Trend OK (EMA{MEM_TREND_EMA_W}W): {memory_debug['mem_trend_ok']}")
print(f"RS vs SMH OK: {memory_debug['mem_rs_ok']}  | RS value: {memory_debug['mem_rs_value']:+.4f}")
print(f"Momentum OK (ROC 3M): {memory_debug['mem_mom_ok']} | ROC: {memory_debug['mem_roc_3m']:+.4f}")
print(f"Shock trigger: {memory_debug['mem_shock']}")
print(f"Memory weight cap (base, before RiskBudget scaling): {memory_debug['mem_weight_cap_base']:.3f}")
line()

print("===== RESULTS =====")
print(f"Strategy CAGR: {fmt_pct(summary['cagr'])} | MaxDD: {fmt_pct(summary['maxdd'])} | Vol: {fmt_pct(summary['vol'])} | Sharpe: {summary['sharpe']:.2f} | Sortino: {summary['sortino']:.2f}")
print(f"QQQ B&H  CAGR: {fmt_pct(cagr(eq_qqq))} | MaxDD: {fmt_pct(max_drawdown(eq_qqq))}")
print(f"SPY B&H  CAGR: {fmt_pct(cagr(eq_spy))} | MaxDD: {fmt_pct(max_drawdown(eq_spy))}")
print(f"Turnover (approx): {summary['turnover_annual']:.2f}x | Avg DEF: {summary['avg_def_weight']:.2f} | Avg RISK: {summary['avg_risk_weight']:.2f} | Churn flags: {summary['churn_flags']}")
line()

print("===== LATEST EXEC WEIGHTS =====")
lw = pd.Series(latest_weights).sort_values(ascending=False)
lw = lw[lw > 1e-6]
for k, v in lw.items():
    print(f"{k:>6}: {v:6.3f}")
line()
print("Outputs written to /outputs (summary.json, macro_debug.json, memory_debug.json, recent_exec_weights.csv, latest_exec_weights.json)")
