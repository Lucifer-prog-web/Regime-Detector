# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 22:48:33 2026

@author: Vidit Jain
"""

# -*- coding: utf-8 -*-
"""
Market Regime Detection for Nifty 50 — FULL PIPELINE (Phases 1–7)
10-Regime Emotional Cycle:
  Bull phase  → Optimism → Enthusiasm → Exhilaration → Euphoria
  Turning     → Unease → Denial
  Bear phase  → Pessimism → Panic → Despair → Capitulation
  Recovery    → Hope → Relief → back to Optimism
Author: Vidit Jain
"""

import requests
import pandas as pd
import yfinance as yf
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
import warnings


warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans     
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment
from scipy.optimize import minimize as _scipy_minimize
from scipy.stats import ttest_1samp as _ttest_1samp
from scipy.stats import skew as _skew, kurtosis as _kurtosis
from sklearn.metrics import calinski_harabasz_score as _ch_score


# ─────────────────────────────────────────────────────────────────────────────
# RISK MANAGEMENT PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
DD_STOP_THRESHOLD  = -0.15
DD_RESUME_THRESHOLD = -0.05
VOL_SCALE_WINDOW   = 21
VOL_TARGET_WINDOW  = 252
MAX_WEIGHT         = 1.0
N_BOOTSTRAP        = 1000

# ── Optimizer parameters (used in optimize_regime_weights and charts) ─────────
OPT_LAMBDA_CAGR  = 0.5    # weight on CAGR term in objective
OPT_LAMBDA_DD    = 0.3    # penalty on MaxDrawdown
OPT_REG_LAMBDA   = 0.05   # L2 regularization toward prior weights
OPT_MIN_EXPOSURE = 0.40   # average portfolio exposure floor
OPT_BULL_MIN_W   = 0.50   # minimum weight for bull regimes
OPT_BULL_REGIMES = {"Optimism", "Enthusiasm", "Exhilaration", "Hope"}

# Try to import hmmlearn, make HMM optional
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
    print("HMM enabled (hmmlearn found)")
except ImportError:
    HMM_AVAILABLE = False
    print("HMM disabled (hmmlearn not installed — using GMM as primary model)")
    
# ── HSMM: uses hmmlearn + scipy (already available if HMM_AVAILABLE) ─────────
try:
    from scipy.stats import nbinom as _nbinom
    from scipy.special import gammaln as _gammaln
    HSMM_AVAILABLE = HMM_AVAILABLE   # HSMM shares the hmmlearn dependency
    if HSMM_AVAILABLE:
        print("HSMM enabled (hmmlearn + scipy found)")
    else:
        print("HSMM disabled (hmmlearn not installed)")
except ImportError:
    HSMM_AVAILABLE = False
    print("HSMM disabled (scipy.stats.nbinom not found)")
    
from regime_transitions import (
    REGIME_TRANSITION_THRESHOLDS,
    REGIME_COLORS,
    REGIME_SEQUENCE,
    REGIME_PHASE,
    check_transition_signal,
    get_transition_summary,
    scan_all_transitions,
    resolve_regime,            # ← add this
)

# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_YEARS   = 4          # rolling training window (years)
TEST_YEARS    = 1          # out-of-sample test window (years)
TRANS_COST    = 0.0005     # one-way transaction cost (5 bps, ~0.05%)
RESOLVE_WINDOW = 5         # rolling window for resolve_regime / scan_all_transitions

MIN_REGIME_DURATION = 10   # minimum days before a regime switch is accepted
FEAT_COLS_CORE = [         # features used in ALL model fitting (train + test)
    "Market_Return", "Volatility", "Dispersion", "Avg_Correlation",
    "Breadth", "Momentum", "RSI",
    # enhanced features added in Phase 2:
    "Rolling_Skew", "Rolling_Kurt", "Drawdown_Depth",
    "Vol_Pct_252", "Vol_Ratio", "RealVol_20", "RealVol_60",
    "Price_vs_50DMA", "Price_vs_200DMA", "Slope_50DMA",
    "Momentum_3M", "Momentum_6M", "ATR_Pct", "Range_Expansion",
]

# ─────────────────────────────────────────────────────────────────────────────
# REGIME DEFINITIONS  (optimal k selected per walk-forward window by BIC/silhouette)
# ─────────────────────────────────────────────────────────────────────────────

K_MIN = 3          # minimum regimes to evaluate
K_MAX = 15         # maximum regimes to evaluate
BIC_TOLERANCE = 0.02   # a model with BIC within 2% of best is considered "close"

from sklearn.metrics import silhouette_score as _silhouette_score




# Color palette mapping each emotional state
'''REGIME_COLORS = {
    "Optimism":     "#27ae60",   # green
    "Enthusiasm":   "#2ecc71",   # light green
    "Exhilaration": "#f1c40f",   # yellow-green
    "Euphoria":     "#e67e22",   # orange  (peak — danger zone)
    "Unease":       "#e74c3c",   # red-orange
    "Denial":       "#c0392b",   # red
    "Pessimism":    "#8e44ad",   # purple
    "Despair":      "#2c3e50",   # dark navy (trough)
    "Capitulation": "#7f8c8d",   # grey
    "Hope":         "#3498db",   # blue
}

# Characteristics per regime used for auto-labelling
#  Each entry: (min_return_rank_frac, max_return_rank_frac,
#               min_vol_rank_frac,    max_vol_rank_frac,
#               min_breadth,          label)
#  These are heuristic thresholds on normalised ranks (0=lowest, 1=highest)
REGIME_SEQUENCE = [
    "Optimism", "Enthusiasm", "Exhilaration", "Euphoria",
    "Unease", "Denial",
    "Pessimism", "Despair", "Capitulation",
    "Hope",
]'''

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — DATA COLLECTION
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("PHASE 1 — Data Collection")
print("=" * 60)

headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/"
}

session = requests.Session()
session.get("https://www.nseindia.com", headers=headers)
url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"

try:
    response = session.get(url, headers=headers, timeout=10)
    data = response.json()
    df_nse = pd.DataFrame(data["data"])
    stocks = [s + ".NS" for s in df_nse["symbol"].tolist()]
    print(f"Downloaded {len(stocks)} stocks from NSE API")
except Exception as e:
    print(f"NSE API failed ({e}), using fallback Nifty 50 list")
    # Fallback list of Nifty 50 stocks
    stocks = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
        "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
        "TITAN.NS", "BAJFINANCE.NS", "DMART.NS", "WIPRO.NS", "HCLTECH.NS",
        "ULTRACEMCO.NS", "NTPC.NS", "POWERGRID.NS", "TATAMOTORS.NS", "TATASTEEL.NS",
        "ONGC.NS", "JSWSTEEL.NS", "HINDALCO.NS", "BAJAJFINSV.NS", "ADANIENT.NS",
        "COALINDIA.NS", "GRASIM.NS", "SBILIFE.NS", "INDUSINDBK.NS", "DRREDDY.NS",
        "CIPLA.NS", "BRITANNIA.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "NESTLEIND.NS",
        "TECHM.NS", "BPCL.NS", "TATACONSUM.NS", "APOLLOHOSP.NS", "HDFCLIFE.NS",
        "SHRIRAMFIN.NS", "M&M.NS", "ADANIPORTS.NS", "LTIM.NS", "PIDILITIND.NS"
    ]

start = dt.datetime.today() - dt.timedelta(days=15 * 365)
end   = dt.datetime.today()

print(f"Downloading {len(stocks)} stocks | {start.date()} → {end.date()} ...")
raw   = yf.download(stocks, start=start, end=end, progress=False)
close = raw["Close"].copy()
close = close.loc[:, ~close.columns.duplicated()]
close = close.loc[:, close.isna().mean() < 0.20].ffill().bfill()

log_ret = np.log(close / close.shift(1)).dropna(how="all")
print(f"Log-return matrix: {log_ret.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 2 — Feature Engineering")
print("=" * 60)

market_return = log_ret.mean(axis=1);        market_return.name = "Market_Return"
market_vol    = market_return.rolling(20).std(); market_vol.name = "Volatility"
dispersion    = log_ret.std(axis=1);          dispersion.name    = "Dispersion"

def rolling_avg_corr(returns_df, window=30):
    dates    = returns_df.index
    n_cols   = returns_df.shape[1]
    avg_corr = pd.Series(index=dates, dtype=float, name="Avg_Correlation")
    upper    = np.triu_indices(n_cols, k=1)
    for i in range(window - 1, len(dates)):
        avg_corr.iloc[i] = returns_df.iloc[i-window+1:i+1].corr().values[upper].mean()
    return avg_corr

print("Computing rolling correlation (≈30 s) ...")
avg_corr = rolling_avg_corr(log_ret, window=30)
breadth  = (log_ret > 0).mean(axis=1); breadth.name = "Breadth"

# ── Momentum ──────────────────────────────────────────────────────────────────
momentum    = market_return.rolling(10).sum();  momentum.name    = "Momentum"
momentum_3m = market_return.rolling(63).sum();  momentum_3m.name = "Momentum_3M"
momentum_6m = market_return.rolling(126).sum(); momentum_6m.name = "Momentum_6M"

# ── RSI ───────────────────────────────────────────────────────────────────────
def compute_rsi(series, window=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

rsi = compute_rsi(market_return, 14); rsi.name = "RSI"

# ── Market structure features ─────────────────────────────────────────────────
rolling_skew = (market_return.rolling(20)
                .apply(lambda x: float(_skew(x)), raw=True));  rolling_skew.name = "Rolling_Skew"
rolling_kurt = (market_return.rolling(20)
                .apply(lambda x: float(_kurtosis(x)), raw=True)); rolling_kurt.name = "Rolling_Kurt"

# Drawdown depth: current level vs rolling 252-day high (using log-cumsum proxy)
cum_log   = market_return.cumsum()
roll_peak = cum_log.rolling(252, min_periods=1).max()
drawdown_depth = (cum_log - roll_peak); drawdown_depth.name = "Drawdown_Depth"

# ATR-style: high-low range as fraction of level
# Approximate using daily return range across stocks
daily_hi  = log_ret.max(axis=1)
daily_lo  = log_ret.min(axis=1)
atr_pct   = (daily_hi - daily_lo).rolling(14).mean(); atr_pct.name = "ATR_Pct"

# Range expansion: today's range vs 20-day average range
daily_range     = daily_hi - daily_lo
range_expansion = daily_range / daily_range.rolling(20).mean().replace(0, np.nan)
range_expansion.name = "Range_Expansion"

# Gap frequency: fraction of days with |return| > 1 std dev (last 20 days)
roll_std_20      = market_return.rolling(20).std()
gap_series       = (market_return.abs() > roll_std_20).astype(float)
# (stored as ATR_Pct already captures similar info; skip separate col to avoid bloat)

# ── Volatility regime features ────────────────────────────────────────────────
real_vol_20 = market_return.rolling(20).std() * np.sqrt(252); real_vol_20.name = "RealVol_20"
real_vol_60 = market_return.rolling(60).std() * np.sqrt(252); real_vol_60.name = "RealVol_60"

# Volatility percentile over 252-day window
vol_pct_252 = (market_vol.rolling(252, min_periods=60)
               .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False))
vol_pct_252.name = "Vol_Pct_252"

# Short/long vol ratio (regime transition signal)
short_vol  = market_return.rolling(10).std()
long_vol   = market_return.rolling(60).std()
vol_ratio  = (short_vol / long_vol.replace(0, np.nan)).clip(0.1, 5.0)
vol_ratio.name = "Vol_Ratio"

# ── Trend / DMA features ─────────────────────────────────────────────────────
# Build a Nifty proxy from equal-weight portfolio cumulative log return
nifty_proxy   = np.exp(market_return.cumsum())   # rebased to 1.0 at start
dma_50        = nifty_proxy.rolling(50).mean()
dma_200       = nifty_proxy.rolling(200).mean()

price_vs_50   = (nifty_proxy / dma_50.replace(0, np.nan) - 1);   price_vs_50.name  = "Price_vs_50DMA"
price_vs_200  = (nifty_proxy / dma_200.replace(0, np.nan) - 1);  price_vs_200.name = "Price_vs_200DMA"

# Slope of 50 DMA: percentage change over last 10 days
slope_50      = dma_50.pct_change(10).fillna(0); slope_50.name = "Slope_50DMA"

# ── Assemble full feature matrix ──────────────────────────────────────────────
features = pd.concat([
    market_return, market_vol, dispersion, avg_corr, breadth,
    momentum, rsi,
    # Enhanced features:
    rolling_skew, rolling_kurt, drawdown_depth,
    vol_pct_252, vol_ratio, real_vol_20, real_vol_60,
    price_vs_50, price_vs_200, slope_50,
    momentum_3m, momentum_6m, atr_pct, range_expansion,
], axis=1).dropna()

print(f"Feature matrix: {features.shape}  "
      f"({features.shape[1]} features, {features.shape[0]} days)")
print(features.describe().round(4))

def select_optimal_regime_count(X_train: np.ndarray,
                                 k_min: int = K_MIN,
                                 k_max: int = K_MAX,
                                 bic_tol: float = BIC_TOLERANCE) -> tuple:
    """
    Evaluates GMM BIC and silhouette score for k = k_min..k_max on X_train.
    Selection rule (transparent, no composite score):
      1. Find the k with the lowest BIC.
      2. Collect all k whose BIC is within bic_tol (2%) of the minimum — these
         are statistically indistinguishable by BIC.
      3. Among those candidates, pick the one with the highest silhouette score.
      4. If still tied, prefer the smaller k (parsimony).

    Returns
    -------
    best_k   : int
    bic_vals : dict {k: bic}
    sil_vals : dict {k: silhouette}
    """
    bic_vals = {}
    sil_vals = {}
    converged = {}

    print(f"  [select_k] Scanning k={k_min}..{k_max} on {len(X_train)} training rows ...")
    for k in range(k_min, k_max + 1):
        try:
            gm = GaussianMixture(n_components=k, covariance_type="full",
                                  random_state=42, n_init=5, max_iter=300)
            gm.fit(X_train)
            bic_vals[k]   = gm.bic(X_train)
            converged[k]  = gm.converged_

            labels = gm.predict(X_train)
            # silhouette requires at least 2 unique labels
            if len(np.unique(labels)) >= 2:
                sil_vals[k] = _silhouette_score(X_train, labels,
                                                 sample_size=min(2000, len(X_train)),
                                                 random_state=42)
            else:
                sil_vals[k] = -1.0

        except Exception as ex:
            print(f"    k={k} failed: {ex}")
            bic_vals[k] = np.inf
            sil_vals[k] = -1.0
            converged[k] = False

    # ── Selection rule ──────────────────────────────────────────────────────
    best_bic    = min(bic_vals.values())
    bic_thresh  = best_bic * (1 + bic_tol) if best_bic > 0 else best_bic + abs(best_bic) * bic_tol

    # All k within BIC tolerance of the minimum
    candidates  = [k for k, b in bic_vals.items()
                   if b <= bic_thresh and converged.get(k, False)]
    if not candidates:
        # fallback: just use lowest BIC k, even if not converged
        candidates = [min(bic_vals, key=bic_vals.get)]

    # Among candidates, pick highest silhouette; ties broken by smaller k
    best_k = max(candidates, key=lambda k: (sil_vals.get(k, -1), -k))

    print(f"  [select_k] Best k={best_k}  "
          f"BIC={bic_vals[best_k]:,.1f}  "
          f"Silhouette={sil_vals.get(best_k, float('nan')):.4f}  "
          f"Candidates={candidates}")

    return best_k, bic_vals, sil_vals

# ─────────────────────────────────────────────────────────────────────────────
# REGIME ACCURACY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def enforce_min_duration(label_series: pd.Series,
                          min_days: int = MIN_REGIME_DURATION) -> pd.Series:
    """
    Persistence filter: suppresses regime switches that last fewer than
    min_days. A short-lived new regime is replaced with the prior regime
    until it has persisted for at least min_days consecutively.

    No look-ahead: decisions use only labels up to current day.
    Applied AFTER resolve_regime, before strategy weights.
    """
    labels  = label_series.values.tolist()
    out     = list(labels)
    n       = len(labels)
    i       = 0
    while i < n:
        current = out[i]
        run_end = i
        while run_end + 1 < n and out[run_end + 1] == current:
            run_end += 1
        run_len = run_end - i + 1
        if run_len < min_days and i > 0:
            # Replace with previous regime
            prev = out[i - 1]
            for j in range(i, run_end + 1):
                out[j] = prev
        i = run_end + 1

    return pd.Series(out, index=label_series.index, name=label_series.name)


def compute_regime_prob_diagnostics(prob_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame of per-state probabilities (columns = states),
    returns top_prob, entropy, and second_best_prob per row.
    No composite score — raw probability statistics only.
    """
    import scipy.stats as _sp_stats
    top_prob    = prob_df.max(axis=1)
    second_prob = prob_df.apply(lambda r: r.nlargest(2).iloc[-1]
                                 if len(r) >= 2 else 0.0, axis=1)
    entropy     = prob_df.apply(
        lambda r: float(_sp_stats.entropy(r.clip(lower=1e-10))), axis=1
    )
    return pd.DataFrame({
        "top_regime_probability" : top_prob,
        "second_best_probability": second_prob,
        "entropy"                : entropy,
    }, index=prob_df.index)


def save_enhanced_predictions(test_df: pd.DataFrame,
                               hmm_prob_cols: list,
                               window_label: str = "") -> None:
    """Appends per-day enhanced regime predictions to the global collector."""
    prob_diag = pd.DataFrame(index=test_df.index)
    if hmm_prob_cols:
        prob_df   = test_df[[c for c in hmm_prob_cols if c in test_df.columns]]
        prob_diag = compute_regime_prob_diagnostics(prob_df)

    for date in test_df.index:
        rl = test_df.loc[date, "Resolved_Label"] if "Resolved_Label" in test_df.columns else ""
        ri = test_df.loc[date, "HMM_Regime"]     if "HMM_Regime"     in test_df.columns else ""
        ri_hsmm = test_df.loc[date, "HSMM_Regime"] if "HSMM_Regime" in test_df.columns else ""
        tp = float(prob_diag.loc[date, "top_regime_probability"])  \
             if date in prob_diag.index else np.nan
        en = float(prob_diag.loc[date, "entropy"])                 \
             if date in prob_diag.index else np.nan
        sp = float(prob_diag.loc[date, "second_best_probability"]) \
             if date in prob_diag.index else np.nan
        _enhanced_pred_rows.append({
            "date"               : str(date.date()),
            "predicted_regime"   : rl,
            "regime_probability" : round(tp, 6) if not np.isnan(tp) else "",
            "entropy"            : round(en, 6) if not np.isnan(en) else "",
            "state_id"           : int(ri)      if ri != "" and not pd.isna(ri) else "",
            "hsmm_state_id"      : int(ri_hsmm) if ri_hsmm != "" and not pd.isna(ri_hsmm) else "",
            "confidence_rank"    : "high" if (not np.isnan(tp) and tp > 0.6) else
                                   "medium" if (not np.isnan(tp) and tp > 0.4) else "low",
            "window"             : window_label,
        })

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — REGIME DETECTION  (data-driven k via BIC + silhouette)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 3 — ML Models  (k selected by BIC + silhouette on full history)")
print("=" * 60)

scaler = StandardScaler()
X      = scaler.fit_transform(features)

# ── Select optimal k on the full training history ──
print("\n[0/3] Selecting optimal k via BIC + silhouette ...")
N_REGIMES, _bic_full, _sil_full = select_optimal_regime_count(X)

# Diagnostic charts
_ks = sorted(_bic_full.keys())
fig_diag, (ax_bic, ax_sil) = plt.subplots(1, 2, figsize=(14, 5))
ax_bic.plot(_ks, [_bic_full[k] for k in _ks], marker="o", color="#2980b9", linewidth=2)
ax_bic.axvline(N_REGIMES, color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"Selected k={N_REGIMES}")
ax_bic.set_xlabel("Number of Regimes (k)"); ax_bic.set_ylabel("BIC (lower = better)")
ax_bic.set_title("GMM BIC vs Cluster Count", fontweight="bold")
ax_bic.legend(); ax_bic.grid(alpha=0.3)

ax_sil.plot(_ks, [_sil_full.get(k, np.nan) for k in _ks],
            marker="s", color="#27ae60", linewidth=2)
ax_sil.axvline(N_REGIMES, color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"Selected k={N_REGIMES}")
ax_sil.set_xlabel("Number of Regimes (k)"); ax_sil.set_ylabel("Silhouette (higher = better)")
ax_sil.set_title("Silhouette Score vs Cluster Count", fontweight="bold")
ax_sil.legend(); ax_sil.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("bic_vs_clusters.png", dpi=150)
plt.savefig("silhouette_vs_clusters.png", dpi=150)
plt.show()
print(f"  Saved: bic_vs_clusters.png  |  silhouette_vs_clusters.png")
print(f"  ► Full-history optimal k = {N_REGIMES}")

# ─────────────────────────────────────────────────────────────────────────────
# HSMM — Hidden Semi-Markov Model (duration-aware Gaussian HMM)
# ─────────────────────────────────────────────────────────────────────────────

class GaussianHSMM:
    """
    Hidden Semi-Markov Model built on top of hmmlearn's GaussianHMM.

    Key difference from HMM:
        - Fits per-state negative-binomial duration distributions from the
          Viterbi path on training data.
        - Applies a duration penalty during Viterbi decoding at inference time,
          discouraging transitions before the expected minimum state duration.

    Parameters
    ----------
    n_components : int
        Number of hidden states (regimes).
    covariance_type : str
        Passed to GaussianHMM ('full' recommended).
    n_iter : int
        EM iterations for the underlying GaussianHMM.
    min_duration : int
        Hard floor — states lasting fewer than this many steps are penalised
        heavily during decoding.
    random_state : int
    """

    def __init__(self, n_components=5, covariance_type="full",
                 n_iter=300, min_duration=5, random_state=42):
        self.n_components    = n_components
        self.covariance_type = covariance_type
        self.n_iter          = n_iter
        self.min_duration    = min_duration
        self.random_state    = random_state

        self._hmm            = None
        self._dur_r          = None   # neg-binom r param per state
        self._dur_p          = None   # neg-binom p param per state
        self._dur_mean       = None   # empirical mean duration per state
        self.converged_      = False

    # ── Fit ───────────────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray):
        """Fit GaussianHMM then estimate duration distributions from Viterbi path."""
        from hmmlearn.hmm import GaussianHMM as _GHMM

        self._hmm = _GHMM(
            n_components    = self.n_components,
            covariance_type = self.covariance_type,
            n_iter          = self.n_iter,
            random_state    = self.random_state,
        )
        self._hmm.fit(X)
        self.converged_ = self._hmm.monitor_.converged

        # Collect run-length statistics from Viterbi path
        viterbi_path = self._hmm.predict(X)
        self._fit_duration_model(viterbi_path)
        return self

    def _fit_duration_model(self, path: np.ndarray):
        """Fit negative-binomial duration parameters per state from Viterbi path."""
        k = self.n_components
        run_lengths = {s: [] for s in range(k)}

        # Extract runs
        i = 0
        n = len(path)
        while i < n:
            s = path[i]
            j = i
            while j < n and path[j] == s:
                j += 1
            run_lengths[s].append(j - i)
            i = j

        self._dur_mean = np.ones(k) * self.min_duration
        self._dur_r    = np.ones(k) * 2.0
        self._dur_p    = np.ones(k) * 0.5

        for s in range(k):
            runs = np.array(run_lengths[s], dtype=float)
            if len(runs) < 3:
                continue
            mu  = runs.mean()
            var = runs.var()
            self._dur_mean[s] = max(mu, 1.0)
            if var > mu:
                # Method-of-moments fit for negative binomial
                r_hat = mu ** 2 / max(var - mu, 1e-6)
                p_hat = r_hat / (r_hat + mu)
                self._dur_r[s] = max(r_hat, 0.1)
                self._dur_p[s] = float(np.clip(p_hat, 0.01, 0.99))
            else:
                # Variance ≤ mean → treat as geometric (r=1)
                self._dur_r[s] = 1.0
                self._dur_p[s] = float(np.clip(1.0 / max(mu, 1.0), 0.01, 0.99))

    # ── Duration log-probability ──────────────────────────────────────────────
    def _log_duration_prob(self, state: int, duration: int) -> float:
        """Log P(duration | state) under fitted negative binomial."""
        r = self._dur_r[state]
        p = self._dur_p[state]
        d = max(duration, 1)
        # log PMF of negative binomial
        lp = (_gammaln(r + d) - _gammaln(r) - _gammaln(d + 1)
              + r * np.log(p + 1e-15)
              + d * np.log(1 - p + 1e-15))
        return float(lp)

    # ── Duration-aware Viterbi predict ────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Viterbi decoding with duration penalty.

        For each time step the log-likelihood comes from:
            HMM emission log-prob  +  duration_penalty(current_run_length, state)

        The duration penalty discourages run lengths that are far below the
        expected minimum, while not blocking long runs.
        """
        if self._hmm is None:
            raise RuntimeError("HSMM not fitted. Call fit() first.")

        n, k = len(X), self.n_components
        # Get HMM log-emission matrix (n_obs × n_states)
        log_emit = self._hmm._compute_log_likelihood(X)   # shape (n, k)

        # Viterbi with state-run tracking
        log_delta = np.full((n, k), -np.inf)
        psi       = np.zeros((n, k), dtype=int)
        run_len   = np.ones((n, k), dtype=int)   # current run length reaching (t, s)

        log_pi   = np.log(self._hmm.startprob_ + 1e-15)
        log_A    = np.log(self._hmm.transmat_   + 1e-15)

        # Initialise t=0
        for s in range(k):
            dur_pen = max(0.0, self._log_duration_prob(s, 1))
            log_delta[0, s] = log_pi[s] + log_emit[0, s] + dur_pen
            run_len[0, s]   = 1

        # Forward pass
        for t in range(1, n):
            for s in range(k):
                best_score = -np.inf
                best_prev  = 0
                best_rl    = 1
                for prev in range(k):
                    if prev == s:
                        # Stay in same state: extend run
                        rl    = run_len[t-1, prev] + 1
                        score = log_delta[t-1, prev] + log_A[prev, s]
                    else:
                        # Transition from prev → s: new run starts at 1
                        rl    = 1
                        score = log_delta[t-1, prev] + log_A[prev, s]

                    # Duration log-probability as an additive reward/penalty
                    dur_lp = self._log_duration_prob(s, rl)
                    score += log_emit[t, s] + dur_lp

                    if score > best_score:
                        best_score = score
                        best_prev  = prev
                        best_rl    = rl

                log_delta[t, s] = best_score
                psi[t, s]       = best_prev
                run_len[t, s]   = best_rl

        # Backtrack
        path    = np.zeros(n, dtype=int)
        path[n-1] = int(np.argmax(log_delta[n-1]))
        for t in range(n-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]

        return path

    # ── Posterior-like soft assignment ────────────────────────────────────────
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns soft regime probabilities.

        Uses HMM posterior (forward-backward) as the base — this is the
        tractable approximation for the HSMM emission model since the
        emission parameters are shared with the underlying GaussianHMM.
        The duration model is applied only during hard Viterbi decoding.
        """
        if self._hmm is None:
            raise RuntimeError("HSMM not fitted. Call fit() first.")
        return self._hmm.predict_proba(X)

# Model 1 — KMeans
print(f"\n[1/3] K-Means  (k={N_REGIMES}) ...")
kmeans        = KMeans(n_clusters=N_REGIMES, random_state=42, n_init=30)
kmeans_labels = kmeans.fit_predict(X)
features["KMeans_Regime"] = kmeans_labels

# Model 2 — GMM
print(f"[2/3] GMM  (k={N_REGIMES}) ...")
gmm       = GaussianMixture(n_components=N_REGIMES, covariance_type="full",
                             random_state=42, n_init=10, max_iter=300)
gmm.fit(X)
gmm_labels = gmm.predict(X)
gmm_probs  = gmm.predict_proba(X)
features["GMM_Regime"] = gmm_labels
gmm_prob_df = pd.DataFrame(gmm_probs, index=features.index,
                            columns=[f"GMM_P_R{i}" for i in range(N_REGIMES)])

# Model 3 — HMM (optional)
if HMM_AVAILABLE:
    print(f"[3/3] HMM  (k={N_REGIMES}) ...")
    hmm_model = GaussianHMM(n_components=N_REGIMES, covariance_type="full",
                             n_iter=300, random_state=42)
    hmm_model.fit(X)
    hmm_labels = hmm_model.predict(X)
    hmm_probs  = hmm_model.predict_proba(X)
    features["HMM_Regime"] = hmm_labels
    hmm_prob_df = pd.DataFrame(hmm_probs, index=features.index,
                                columns=[f"HMM_P_R{i}" for i in range(N_REGIMES)])
else:
    print(f"[3/3] HMM skipped — using GMM as proxy  (k={N_REGIMES})")
    features["HMM_Regime"] = features["GMM_Regime"]
    hmm_prob_df = gmm_prob_df.copy()
    hmm_prob_df.columns = [f"HMM_P_R{i}" for i in range(N_REGIMES)]
    
# Model 4 — HSMM (optional, requires hmmlearn + scipy)
if HSMM_AVAILABLE:
    print(f"[4/4] HSMM  (k={N_REGIMES}) ...")
    hsmm_model = GaussianHSMM(n_components=N_REGIMES, covariance_type="full",
                               n_iter=300, min_duration=MIN_REGIME_DURATION,
                               random_state=42)
    hsmm_model.fit(X)
    hsmm_labels = hsmm_model.predict(X)
    hsmm_probs  = hsmm_model.predict_proba(X)
    features["HSMM_Regime"] = hsmm_labels
    hsmm_prob_df = pd.DataFrame(hsmm_probs, index=features.index,
                                 columns=[f"HSMM_P_R{i}" for i in range(N_REGIMES)])
    if hsmm_model.converged_:
        print(f"  [HSMM] Converged. Mean state durations: "
              + ", ".join([f"S{s}={hsmm_model._dur_mean[s]:.1f}d"
                           for s in range(N_REGIMES)]))
    else:
        print("  [HSMM] Warning: did not converge — increase n_iter")
else:
    print(f"[4/4] HSMM skipped — using HMM as proxy  (k={N_REGIMES})")
    features["HSMM_Regime"] = features["HMM_Regime"]
    hsmm_prob_df = hmm_prob_df.copy()
    hsmm_prob_df.columns = [f"HSMM_P_R{i}" for i in range(N_REGIMES)]

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — EMOTIONAL CYCLE LABELLING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 4 — Emotional Cycle Labelling")
print("=" * 60)


# ── Emotion sequence ordered from worst to best market conditions ──────────
# This ordering is the ONLY fixed assumption — it matches the Wall Street
# emotional cycle and is used purely as a positional sequence, not as
# hand-picked decimal values.
EMOTION_SEQUENCE_ORDERED = [
    "Despair",       # rank 0  — lowest return, highest fear
    "Capitulation",  # rank 1
    "Pessimism",     # rank 2
    "Denial",        # rank 3
    "Unease",        # rank 4
    "Hope",          # rank 5
    "Optimism",      # rank 6
    "Enthusiasm",    # rank 7
    "Exhilaration",  # rank 8
    "Euphoria",      # rank 9  — highest return, elevated vol
]


def build_data_driven_emotion_mapping(feature_df: pd.DataFrame,
                                       cluster_col: str,
                                       window_label: str = "") -> dict:
    """
    Derives cluster → emotion mapping entirely from actual cluster statistics.
    No hand-picked decimal fingerprints.

    Algorithm (transparent, two-step):
    ─────────────────────────────────
    Step 1 — Sort clusters by (avg_return ASC, avg_volatility DESC).
        Primary key  : average Market_Return — lowest return = most fearful
        Tie-break key: average Volatility    — for similar returns, higher
                       vol = more panicked state (Despair before Capitulation)
        Special rule : the cluster with the highest return AND elevated vol
                       (top-return tertile + vol above median) is anchored to
                       Euphoria regardless of sort position, because Euphoria
                       is the only state with high return + high vol.

    Step 2 — Assign emotions from EMOTION_SEQUENCE_ORDERED by position.
        k < 10 : evenly spaced indices into the 10-emotion sequence
        k = 10 : direct 1-to-1 positional assignment
        k > 10 : 10 anchor emotions assigned first, overflow get suffix
    """
    feat_cols = [c for c in ["Market_Return", "Volatility", "Momentum", "RSI", "Breadth"]
                 if c in feature_df.columns]

    summary = feature_df.groupby(cluster_col)[feat_cols].mean()
    clusters = summary.index.tolist()
    k = len(clusters)

    ret_col = "Market_Return"
    vol_col = "Volatility"

    # ── Special Euphoria anchor ──────────────────────────────────────────────
    # Euphoria = highest return AND vol above median (rising vol at the peak)
    vol_median  = summary[vol_col].median()
    ret_max_idx = summary[ret_col].idxmax()
    euphoria_anchor = None
    if summary.loc[ret_max_idx, vol_col] >= vol_median:
        euphoria_anchor = ret_max_idx   # qualifies as Euphoria

    # ── Sort remaining clusters by return ASC, vol DESC (tie-break) ─────────
    remaining = [c for c in clusters if c != euphoria_anchor]
    remaining_sorted = sorted(
        remaining,
        key=lambda c: (summary.loc[c, ret_col], -summary.loc[c, vol_col])
    )

    # Build ordered cluster list: sorted first, Euphoria anchored at the end
    if euphoria_anchor is not None:
        ordered_clusters = remaining_sorted + [euphoria_anchor]
    else:
        ordered_clusters = remaining_sorted   # no Euphoria anchor; pure return sort

    # ── Assign emotion labels by position ───────────────────────────────────
    label_map = {}
    n_emotions = len(EMOTION_SEQUENCE_ORDERED)

    if k == n_emotions:
        # Direct 1-to-1
        for cluster, emotion in zip(ordered_clusters, EMOTION_SEQUENCE_ORDERED):
            label_map[cluster] = emotion

    elif k < n_emotions:
        # Pick evenly-spaced emotions from the sequence to preserve cycle shape
        # e.g. k=5: indices [0, 2, 4, 6, 9] → Despair, Pessimism, Unease,
        #           Optimism, Euphoria
        indices = [round(i * (n_emotions - 1) / (k - 1)) for i in range(k)] if k > 1 \
                  else [n_emotions // 2]
        # Always anchor index 0 → lowest emotion, index -1 → highest emotion
        indices[0]  = 0
        indices[-1] = n_emotions - 1
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for idx in indices:
            while idx in seen:
                idx = min(idx + 1, n_emotions - 1)
            seen.add(idx)
            deduped.append(idx)
        for cluster, idx in zip(ordered_clusters, deduped):
            label_map[cluster] = EMOTION_SEQUENCE_ORDERED[idx]

    else:
        # k > 10: assign 10 anchor clusters first, overflow get suffixes
        anchor_step = k / n_emotions
        anchor_positions = [int(round(i * anchor_step)) for i in range(n_emotions)]
        anchor_positions = list(dict.fromkeys(anchor_positions))  # dedup, keep order
        # Ensure last cluster gets Euphoria
        if ordered_clusters[-1] not in [ordered_clusters[p] for p in anchor_positions
                                         if p < len(ordered_clusters)]:
            anchor_positions[-1] = len(ordered_clusters) - 1

        anchor_clusters = [ordered_clusters[p] for p in anchor_positions
                           if p < len(ordered_clusters)]
        for cluster, emotion in zip(anchor_clusters, EMOTION_SEQUENCE_ORDERED):
            label_map[cluster] = emotion

        # Overflow: each non-anchor cluster gets nearest anchor's emotion + suffix
        suffix_count = {}
        for cluster in ordered_clusters:
            if cluster not in label_map:
                # Find nearest anchor by return distance
                nearest = min(anchor_clusters,
                              key=lambda ac: abs(summary.loc[cluster, ret_col]
                                                 - summary.loc[ac, ret_col]))
                base = label_map[nearest]
                suffix_count[base] = suffix_count.get(base, 1) + 1
                label_map[cluster] = f"{base}_{suffix_count[base]}"

    # ── Diagnostic print ─────────────────────────────────────────────────────
    assigned_emotions = [label_map[c] for c in ordered_clusters]
    print(f"\n  {cluster_col} → data-driven emotional labels (k={k}"
          + (f", window={window_label}" if window_label else "") + "):")
    for cluster in ordered_clusters:
        s = summary.loc[cluster]
        ret_str = f"ret={s[ret_col]:+.5f}" if ret_col in s else ""
        vol_str = f"vol={s[vol_col]:.5f}"  if vol_col in s else ""
        anchor  = " ← Euphoria anchor" if cluster == euphoria_anchor else ""
        print(f"    Cluster {cluster:2d} → {label_map[cluster]:18s}  "
              f"{ret_str}  {vol_str}{anchor}")

    return label_map


def test_mapping_stability(feature_df: pd.DataFrame,
                            cluster_col: str,
                            n_bootstrap: int = 10,
                            flip_threshold: float = 0.30) -> None:
    """
    Bootstrap stability check: resample rows with replacement n_bootstrap times,
    recompute the data-driven mapping, and measure how often each cluster's
    assigned emotion changes vs the baseline mapping.

    Prints a warning if any cluster flips emotion more than flip_threshold
    fraction of the time.
    """
    baseline = build_data_driven_emotion_mapping(feature_df, cluster_col,
                                                  window_label="baseline")
    clusters  = list(baseline.keys())
    flip_counts = {c: 0 for c in clusters}

    for trial in range(n_bootstrap):
        sample = feature_df.sample(frac=1.0, replace=True, random_state=trial)
        trial_map = build_data_driven_emotion_mapping(sample, cluster_col,
                                                       window_label=f"bs_{trial}")
        for c in clusters:
            if trial_map.get(c) != baseline.get(c):
                flip_counts[c] += 1

    print(f"\n  [stability] Bootstrap results ({n_bootstrap} trials):")
    any_unstable = False
    for c in clusters:
        flip_rate = flip_counts[c] / n_bootstrap
        status    = "⚠ UNSTABLE" if flip_rate > flip_threshold else "✓ stable"
        print(f"    Cluster {c:2d} → {baseline[c]:18s}  "
              f"flip_rate={flip_rate:.0%}  {status}")
        if flip_rate > flip_threshold:
            any_unstable = True

    if any_unstable:
        print(f"\n  ⚠ WARNING: Some clusters have unstable emotion assignments "
              f"(flip rate > {flip_threshold:.0%}). "
              f"Consider using a wider training window or fewer regimes.")
    else:
        print(f"\n  ✓ All cluster emotion assignments are stable "
              f"(flip rate ≤ {flip_threshold:.0%} in all cases).")


def label_emotional_cycle(feature_df: pd.DataFrame,
                           regime_col: str,
                           window_label: str = "",
                           run_stability: bool = False) -> dict:
    """
    Public entry point — wraps build_data_driven_emotion_mapping.
    Replaces the old fingerprint-based label_emotional_cycle.
    Signature is backward-compatible: same inputs, same output (label_map dict).
    """
    label_map = build_data_driven_emotion_mapping(
        feature_df, regime_col, window_label=window_label
    )

    if run_stability:
        test_mapping_stability(feature_df, regime_col)

    return label_map

kmeans_label_map = label_emotional_cycle(features, "KMeans_Regime",
                                          window_label="full_history")
gmm_label_map    = label_emotional_cycle(features, "GMM_Regime",
                                          window_label="full_history")
hmm_label_map    = label_emotional_cycle(features, "HMM_Regime",
                                          window_label="full_history",
                                          run_stability=True)
hsmm_label_map   = label_emotional_cycle(features, "HSMM_Regime",
                                          window_label="full_history")

features["KMeans_Label"] = features["KMeans_Regime"].map(kmeans_label_map)
features["GMM_Label"]    = features["GMM_Regime"].map(gmm_label_map)
features["HMM_Label"]    = features["HMM_Regime"].map(hmm_label_map)
features["HSMM_Label"]   = features["HSMM_Regime"].map(hsmm_label_map)

# ── Save cluster_emotion_mapping.csv ────────────────────────────────────────
_mapping_rows = []
_summary_hmm  = features.groupby("HMM_Regime")[
    ["Market_Return", "Volatility"]].mean()
for cluster_id, emotion in hmm_label_map.items():
    _mapping_rows.append({
        "window_start"    : str(features.index[0].date()),
        "cluster_id"      : cluster_id,
        "avg_return"      : round(_summary_hmm.loc[cluster_id, "Market_Return"], 6),
        "avg_volatility"  : round(_summary_hmm.loc[cluster_id, "Volatility"],    6),
        "assigned_emotion": emotion,
    })
_mapping_df = pd.DataFrame(_mapping_rows).sort_values("avg_return")
_mapping_df.to_csv("cluster_emotion_mapping.csv", index=False)
print(f"\n  Saved: cluster_emotion_mapping.csv")
print(_mapping_df.to_string(index=False))

# ── cluster_emotion_profile.png ─────────────────────────────────────────────
_fig_p, _ax_p = plt.subplots(figsize=(12, 6))
_colors_bar = [REGIME_COLORS.get(e, "#888") for e in _mapping_df["assigned_emotion"]]
_x = range(len(_mapping_df))
_bars = _ax_p.bar(_x, _mapping_df["avg_return"] * 100, color=_colors_bar,
                   alpha=0.85, edgecolor="white", linewidth=0.8)
# Overlay volatility as a line on secondary axis
_ax2_p = _ax_p.twinx()
_ax2_p.plot(list(_x), _mapping_df["avg_volatility"] * 100,
            color="#2c3e50", marker="o", linewidth=2, linestyle="--",
            label="Avg Volatility")
_ax2_p.set_ylabel("Avg Volatility (%)", color="#2c3e50")

_ax_p.set_xticks(list(_x))
_ax_p.set_xticklabels(_mapping_df["assigned_emotion"], rotation=40,
                       ha="right", fontsize=9)
_ax_p.set_ylabel("Avg Daily Return (%)")
_ax_p.set_title("Data-Driven Cluster → Emotion Mapping  (HMM, full history)",
                fontsize=13, fontweight="bold")
_ax_p.axhline(0, color="grey", linewidth=0.8, linestyle=":")
_ax_p.grid(alpha=0.25, axis="y")
_ax2_p.legend(loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig("cluster_emotion_profile.png", dpi=150)
plt.show()
print("  Saved: cluster_emotion_profile.png")

master = pd.concat([features, gmm_prob_df, hmm_prob_df, hsmm_prob_df], axis=1)

# After Phase 4, add this one line:
master["Resolved_Label"] = resolve_regime(master, label_col="HMM_Label", window = 5)

# fill warmup NaNs with the first valid label
master["Resolved_Label"] = master["Resolved_Label"].ffill().bfill()

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 — VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 5 — Visualisations")
print("=" * 60)

nifty_proxy = close.reindex(features.index).mean(axis=1)

# 5.1 — Regime Timeline (HMM)
def plot_timeline(price, regime_labels, model_name, color_map):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(price, color="#2c3e50", linewidth=1, label="Nifty 50 proxy", zorder=5)
    prev, start_d = None, None

    for date, regime in regime_labels.items():
        if regime != prev:
            if prev is not None:
                ax1.axvspan(start_d, date, alpha=0.25,
                            color=color_map.get(prev, "grey"), linewidth=0)
            prev, start_d = regime, date
    if prev:
        ax1.axvspan(start_d, regime_labels.index[-1], alpha=0.25,
                    color=color_map.get(prev, "grey"), linewidth=0)

    patches = [mpatches.Patch(color=c, label=r, alpha=0.8)
               for r, c in color_map.items() if r in regime_labels.values]
    ax1.legend(handles=patches, loc="upper left", fontsize=8, ncol=5)
    ax1.set_title(f"{model_name} — Emotional Cycle Regime Timeline (10 States)",
                  fontsize=14, fontweight="bold")
    ax1.set_ylabel("Index Level")
    ax1.grid(alpha=0.3)

    # Emotion cycle bar
    emotion_order = list(REGIME_COLORS.keys())
    regime_num    = regime_labels.map(
        {e: i for i, e in enumerate(emotion_order)}
    ).fillna(0)
    ax2.bar(regime_labels.index,
            [1] * len(regime_labels),
            color=[color_map.get(r, "grey") for r in regime_labels],
            width=2, linewidth=0)
    ax2.set_yticks([])
    ax2.set_ylabel("Regime")
    ax2.set_xlabel("Date")

    plt.tight_layout()
    plt.savefig(f"regime_timeline_{model_name}.png", dpi=150)
    plt.show()
    print(f"  Saved: regime_timeline_{model_name}.png")

plot_timeline(nifty_proxy, master["HMM_Label"],    "HMM",    REGIME_COLORS)
plot_timeline(nifty_proxy, master["GMM_Label"],    "GMM",    REGIME_COLORS)
plot_timeline(nifty_proxy, master["KMeans_Label"], "KMeans", REGIME_COLORS)
plot_timeline(nifty_proxy, master["HSMM_Label"],   "HSMM",   REGIME_COLORS)

# 5.2 — Emotion Cycle Clock  (polar chart showing time spent in each regime)
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})

emotions  = list(REGIME_COLORS.keys())
counts    = master["Resolved_Label"].value_counts()
durations = [counts.get(e, 0) for e in emotions]
total     = sum(durations) or 1

angles = np.linspace(0, 2 * np.pi, len(emotions), endpoint=False)
widths = [(d / total) * 2 * np.pi for d in durations]

# Draw wedges
starts = np.cumsum([0] + widths[:-1])
for i, (emotion, color) in enumerate(REGIME_COLORS.items()):
    ax.bar(starts[i], 1, width=widths[i], bottom=0.2,
           color=color, alpha=0.8, edgecolor="white", linewidth=1.5,
           align="edge")
    mid_angle = starts[i] + widths[i] / 2
    pct = durations[i] / total * 100
    ax.text(mid_angle, 1.35,
            f"{emotion}\n{pct:.1f}%",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color=color)

ax.set_yticks([])
ax.set_xticks([])
ax.set_title("Resolved_Label — Time Spent in Each Emotional State",
             fontsize=14, fontweight="bold", pad=30)
plt.tight_layout()
plt.savefig("emotion_cycle_clock.png", dpi=150)
plt.show()
print("  Saved: emotion_cycle_clock.png")

# 5.3 — Feature Boxplots by Regime
feature_cols = ["Market_Return", "Volatility", "Dispersion",
                "Avg_Correlation", "Breadth", "Momentum", "RSI"]
emotion_order_plot = [e for e in REGIME_COLORS.keys()
                      if e in master["HMM_Label"].unique()]

fig, axes = plt.subplots(2, 4, figsize=(24, 10))
axes = axes.flatten()

for ax, col in zip(axes, feature_cols):
    data_by_r = [master.loc[master["HMM_Label"] == r, col].dropna().values
                 for r in emotion_order_plot]
    bp = ax.boxplot(data_by_r, patch_artist=True, notch=False)
    for patch, r in zip(bp["boxes"], emotion_order_plot):
        patch.set_facecolor(REGIME_COLORS.get(r, "grey"))
        patch.set_alpha(0.7)
    ax.set_xticklabels(emotion_order_plot, rotation=45, ha="right", fontsize=7)
    ax.set_title(col, fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3)

axes[-1].axis("off")
fig.suptitle("HMM — Feature Distribution Across 10 Emotional Regimes",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("feature_boxplots_10regime.png", dpi=150)
plt.show()
print("  Saved: feature_boxplots_10regime.png")

# 5.4 — Transition Heatmap
hmm_seq      = master["HMM_Label"].values
regimes_u    = [e for e in REGIME_COLORS if e in master["HMM_Label"].unique()]
trans_count  = pd.DataFrame(0, index=regimes_u, columns=regimes_u)
for i in range(len(hmm_seq) - 1):
    if hmm_seq[i] in regimes_u and hmm_seq[i+1] in regimes_u:
        trans_count.loc[hmm_seq[i], hmm_seq[i+1]] += 1

trans_prob = trans_count.div(trans_count.sum(axis=1).replace(0, 1), axis=0)

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(trans_prob.values, cmap="RdYlGn", vmin=0, vmax=1)
plt.colorbar(im, ax=ax, label="Transition probability")
ax.set_xticks(range(len(regimes_u))); ax.set_yticks(range(len(regimes_u)))
ax.set_xticklabels(regimes_u, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(regimes_u, fontsize=9)
ax.set_xlabel("To Regime"); ax.set_ylabel("From Regime")
ax.set_title("Resolved — Emotional Cycle Transition Matrix", fontweight="bold", fontsize=14)
for i in range(len(regimes_u)):
    for j in range(len(regimes_u)):
        v = trans_prob.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                color="white" if v > 0.5 else "black")
plt.tight_layout()
plt.savefig("transition_heatmap_10regime.png", dpi=150)
plt.show()
print("  Saved: transition_heatmap_10regime.png")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6 — BACKTEST
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 6 — Backtest (Emotional Cycle Strategy)")
print("=" * 60)

# ── Legacy fallback weights — only used if optimization fails completely ──────
# These are never applied during walk-forward. The optimizer overwrites them
# per window. They exist solely as a safe default for the Phase 5 full-history
# visualisation path which does not run through the walk-forward engine.
_LEGACY_EMOTION_WEIGHTS = {
    "Optimism":     1.00,
    "Enthusiasm":   1.00,
    "Exhilaration": 1.00,
    "Euphoria":     0.80,
    "Unease":       0.60,
    "Denial":       0.50,
    "Pessimism":    0.30,
    "Despair":      0.60,
    "Capitulation": 0.80,
    "Hope":         1.00,
}

# This will be replaced per window during walk-forward.
# Initialise to legacy values so Phase 5 charts still render.
EMOTION_WEIGHTS = dict(_LEGACY_EMOTION_WEIGHTS)


def _sharpe_from_weights(weight_vec: np.ndarray,
                          regime_seq: pd.Series,
                          market_ret: pd.Series,
                          regimes: list,
                          trans_cost: float = TRANS_COST) -> float:
    """
    Objective: maximize log-wealth (= maximize CAGR directly).
    Returns negative mean log return — minimize() will maximize CAGR.
    """
    w_map   = dict(zip(regimes, weight_vec))
    signal  = regime_seq.shift(1).map(w_map).fillna(0.5)
    prev_s  = signal.shift(1).fillna(0.5)
    tc_drag = (signal - prev_s).abs() * trans_cost
    strat   = signal * market_ret - tc_drag

    # Log-wealth maximization — directly targets CAGR
    log_returns = np.log1p(np.exp(strat) - 1)
    return -np.mean(log_returns)   # negative because minimize()


def optimize_regime_weights(train_df: pd.DataFrame,
                              label_col: str = "Resolved_Label",
                              return_col: str = "Market_Return",
                              trans_cost: float = TRANS_COST) -> dict:
    """
    Optimize one allocation weight per emotional regime on training data only.

    Method : SLSQP (Sequential Least Squares Programming)
    Objective: maximize Sharpe ratio (minimize negative Sharpe)
    Bounds   : each weight ∈ [0.0, 1.0]
    No leakage — called only on train_df inside run_single_window.

    Returns
    -------
    dict  {emotion_name: optimized_weight}
    """
    regime_seq  = train_df[label_col].dropna()
    market_ret  = train_df.loc[regime_seq.index, return_col]
    regimes     = sorted(regime_seq.unique().tolist())
    n           = len(regimes)

    if n == 0:
        return {}

    # Initial guess: legacy weights where available, else 0.5
    x0 = np.array([_LEGACY_EMOTION_WEIGHTS.get(r, 0.5) for r in regimes])

    bounds = [(0.0, 1.0)] * n

    result = _scipy_minimize(
        fun     = _sharpe_from_weights,
        x0      = x0,
        args    = (regime_seq, market_ret, regimes, trans_cost),
        method  = "SLSQP",
        bounds  = bounds,
        options = {"maxiter": 500, "ftol": 1e-9},
    )

    if result.success:
        opt_weights = dict(zip(regimes, np.clip(result.x, 0.0, 1.0)))
        print(f"  [optimize_weights] Converged. Sharpe on train = {-result.fun:.3f}")
    else:
        # Fallback to legacy weights — print warning but do not crash
        print(f"  [optimize_weights] ⚠ Optimizer did not converge ({result.message}). "
              f"Using legacy weights as fallback.")
        opt_weights = {r: _LEGACY_EMOTION_WEIGHTS.get(r, 0.5) for r in regimes}

    print(f"  [optimize_weights] Weights for this window:")
    for r in regimes:
        print(f"    {r:18s} → {opt_weights[r]:.4f}")

    return opt_weights


def run_weight_sensitivity_analysis(weights: dict,
                                     train_df: pd.DataFrame,
                                     label_col: str = "Resolved_Label",
                                     return_col: str = "Market_Return",
                                     perturb: float = 0.20,
                                     fragility_threshold: float = 0.30) -> pd.DataFrame:
    """
    Tests robustness of optimized weights by perturbing each regime weight
    by ±perturb fraction and measuring Sharpe, CAGR, MaxDD.

    If any perturbation causes Sharpe to drop by more than fragility_threshold
    relative to the base, prints a fragility warning.

    Returns sensitivity_df for CSV export.
    """
    regime_seq = train_df[label_col].dropna()
    market_ret = train_df.loc[regime_seq.index, return_col]
    regimes    = sorted(weights.keys())

    def _eval(w_map):
        sig    = regime_seq.shift(1).map(w_map).fillna(0.5)
        prev_s = sig.shift(1).fillna(0.5)
        tc     = (sig - prev_s).abs() * TRANS_COST
        strat  = sig * market_ret - tc
        simple = np.exp(strat) - 1
        cum    = np.exp(strat.sum()) - 1
        n_yrs  = len(simple) / 252
        cagr   = (1 + cum) ** (1 / max(n_yrs, 0.01)) - 1
        sharpe = (simple.mean() / simple.std()) * np.sqrt(252) if simple.std() > 1e-10 else 0.0
        cw     = (1 + simple).cumprod()
        max_dd = ((cw - cw.cummax()) / cw.cummax()).min()
        return sharpe, cagr, max_dd

    base_sharpe, base_cagr, base_dd = _eval(weights)
    rows = []

    for regime in regimes:
        for delta_label, delta in [(f"+{perturb:.0%}", +perturb),
                                    (f"-{perturb:.0%}", -perturb)]:
            perturbed = dict(weights)
            perturbed[regime] = float(np.clip(weights[regime] * (1 + delta), 0.0, 1.0))
            sh, cg, dd = _eval(perturbed)
            rows.append({
                "regime"      : regime,
                "perturbation": delta_label,
                "base_weight" : round(weights[regime], 4),
                "new_weight"  : round(perturbed[regime], 4),
                "sharpe"      : round(sh, 4),
                "cagr"        : round(cg, 6),
                "max_dd"      : round(dd, 6),
            })

    sens_df = pd.DataFrame(rows)

    # Fragility check
    if base_sharpe != 0:
        max_drop = sens_df["sharpe"].apply(
            lambda s: (base_sharpe - s) / abs(base_sharpe)
        ).max()
        if max_drop > fragility_threshold:
            print(f"\n  ⚠ WARNING: Strategy is fragile to weight assumptions  "
                  f"(max Sharpe drop = {max_drop:.1%} from a {perturb:.0%} weight change)")
        else:
            print(f"  ✓ Sensitivity check passed  "
                  f"(max Sharpe drop = {max_drop:.1%} within ±{perturb:.0%} perturbation)")

    print(f"  Base Sharpe={base_sharpe:.3f}  CAGR={base_cagr*100:.2f}%  MaxDD={base_dd*100:.2f}%")
    return sens_df

# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

_window_weight_rows  = []   # collects per-window optimized weights for CSV
_window_sens_results = []   # collects per-window sensitivity results for CSV
_window_diag_rows    = []   # collects per-window optimizer diagnostics for CSV
_enhanced_pred_rows  = []   # enhanced per-day regime predictions

def diagnose_weight_optimizer(opt_weights: dict,
                               train_df: pd.DataFrame,
                               test_df: pd.DataFrame,
                               label_col: str = "Resolved_Label",
                               return_col: str = "Market_Return") -> dict:
    """
    Audit function — evaluates the SLSQP-optimized weights on both the
    training window and the held-out test window.  The results go into
    weight_optimizer_diagnostics.csv for monitoring only; they are NOT
    used to drive allocation (run_adaptive_allocator owns that path).

    Returns a flat dict with:
        window_start, window_end,
        train_sharpe, train_cagr, train_max_dd, avg_exposure,
        test_sharpe,  test_cagr,  test_max_dd
    """
    def _eval_period(df: pd.DataFrame, w_map: dict) -> dict:
        regime_seq = df[label_col].dropna()
        mkt        = df.loc[regime_seq.index, return_col]
        signal     = regime_seq.shift(1).map(w_map).fillna(0.5)
        prev_s     = signal.shift(1).fillna(0.5)
        tc         = (signal - prev_s).abs() * TRANS_COST
        strat      = signal * mkt - tc
        simple     = np.exp(strat) - 1
        n_yrs      = len(simple) / 252
        cum        = np.exp(strat.sum()) - 1
        cagr       = (1 + cum) ** (1 / max(n_yrs, 0.01)) - 1
        sharpe     = (simple.mean() / simple.std()) * np.sqrt(252) \
                     if simple.std() > 1e-10 else 0.0
        cw         = (1 + simple).cumprod()
        max_dd     = ((cw - cw.cummax()) / cw.cummax()).min()
        avg_exp    = signal.mean()
        return {
            "sharpe"      : round(float(sharpe),  4),
            "cagr"        : round(float(cagr),    6),
            "max_dd"      : round(float(max_dd),  6),
            "avg_exposure": round(float(avg_exp), 4),
        }

    train_stats = _eval_period(train_df, opt_weights) if not train_df.empty else {}
    test_stats  = _eval_period(test_df,  opt_weights) if not test_df.empty  else {}

    row = {
        "window_start" : str(train_df.index[0].date())  if not train_df.empty else "",
        "window_end"   : str(train_df.index[-1].date()) if not train_df.empty else "",
        "train_sharpe" : train_stats.get("sharpe",       np.nan),
        "train_cagr"   : train_stats.get("cagr",         np.nan),
        "train_max_dd" : train_stats.get("max_dd",       np.nan),
        "avg_exposure" : train_stats.get("avg_exposure", np.nan),
        "test_sharpe"  : test_stats.get("sharpe",        np.nan),
        "test_cagr"    : test_stats.get("cagr",          np.nan),
        "test_max_dd"  : test_stats.get("max_dd",        np.nan),
    }

    print(f"  [diagnose] Train → Sharpe={row['train_sharpe']:.3f}  "
          f"CAGR={row['train_cagr']*100:.2f}%  MaxDD={row['train_max_dd']*100:.2f}%  "
          f"AvgExp={row['avg_exposure']:.2f}")
    print(f"  [diagnose] Test  → Sharpe={row['test_sharpe']:.3f}  "
          f"CAGR={row['test_cagr']*100:.2f}%  MaxDD={row['test_max_dd']*100:.2f}%")

    return row

# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE ALLOCATION ENGINE  (data-driven, no hand-picked values)
# ─────────────────────────────────────────────────────────────────────────────

# Global collector for forward-stats CSV
_regime_fwd_stats_rows = []
_adaptive_alloc_rows   = []


def estimate_regime_forward_stats(train_df: pd.DataFrame,
                                   label_col:  str = "Resolved_Label",
                                   return_col: str = "Market_Return",
                                   forward_lag: int = 1) -> pd.DataFrame:
    """
    For each regime in the training window, estimates:
        avg_forward_return  — mean of next-day log return
        forward_volatility  — std of next-day log return
        sharpe_like_score   — avg / std * sqrt(252), clipped to [-3, 3]
        observations        — number of days in regime

    Uses only training data. forward_lag=1 → next day's return.
    No leakage: the forward return series is entirely within train_df.
    """
    df = train_df[[label_col, return_col]].copy().dropna()
    df["fwd_return"] = df[return_col].shift(-forward_lag)  # next-day return
    df = df.dropna()  # drop last row(s) where forward return is NaN

    records = []
    for regime, grp in df.groupby(label_col):
        fwd = grp["fwd_return"]
        n   = len(fwd)
        avg = fwd.mean()
        vol = fwd.std()
        sharpe_like = (avg / vol * np.sqrt(252)) if vol > 1e-10 else 0.0
        sharpe_like = float(np.clip(sharpe_like, -3.0, 3.0))
        records.append({
            "regime"             : regime,
            "avg_forward_return" : avg,
            "forward_volatility" : vol,
            "sharpe_like_score"  : sharpe_like,
            "observations"       : n,
        })

    stats_df = pd.DataFrame(records).set_index("regime")
    return stats_df


def compute_regime_weights_from_stats(stats_df: pd.DataFrame,
                                       min_weight: float = 0.0,
                                       max_weight: float = 1.0) -> dict:
    """
    Derives a base allocation weight per regime directly from forward statistics.

    Method (transparent):
        1. Shift sharpe_like_score so all values ≥ 0  (shift by |min| if needed)
        2. Normalize to [min_weight, max_weight]
        3. Regimes with negative expected forward return get weight < 0.5
        4. Regimes with positive expected forward return get weight ≥ 0.5

    No hand-picking. Pure monotonic mapping of data-derived Sharpe.
    """
    scores = stats_df["sharpe_like_score"].copy()

    s_min, s_max = scores.min(), scores.max()
    if abs(s_max - s_min) < 1e-10:
        # All regimes identical — equal weight
        norm = pd.Series(0.5, index=scores.index)
    else:
        norm = (scores - s_min) / (s_max - s_min)   # 0 to 1

    # Map to [min_weight, max_weight]
    weights = min_weight + norm * (max_weight - min_weight)
    weights = weights.clip(min_weight, max_weight)

    return weights.to_dict()


def compute_probability_weighted_exposure(test_row_probs: pd.DataFrame,
                                           cluster_to_label: dict,
                                           label_weights: dict) -> pd.Series:
    """
    For each test day, blends regime weights by model probabilities:
        final_weight_t = Σ_i  P(regime_i at t) × weight(regime_i)

    Inputs
    ------
    test_row_probs  : DataFrame of shape (n_days, n_clusters)
                      columns like HMM_P_R0, HMM_P_R1, ...
    cluster_to_label: {cluster_int: emotion_str}  mapping from this window
    label_weights   : {emotion_str: float}        data-driven base weights

    Returns
    -------
    pd.Series of blended daily weights, index = test_row_probs.index
    """
    n_clusters = test_row_probs.shape[1]
    blended    = pd.Series(0.0, index=test_row_probs.index)

    for col_idx in range(n_clusters):
        col_name = test_row_probs.columns[col_idx]
        prob_series = test_row_probs[col_name]
        emotion = cluster_to_label.get(col_idx)
        if emotion is None:
            continue
        w = label_weights.get(emotion, 0.5)
        blended += prob_series * w

    return blended.clip(0.0, 1.0)


def apply_volatility_targeting(weight_series: pd.Series,
                                market_ret: pd.Series,
                                current_window: int = VOL_SCALE_WINDOW,
                                target_window:  int = VOL_TARGET_WINDOW,
                                max_cap: float  = MAX_WEIGHT) -> pd.Series:
    """
    Scales weights by (target_vol / current_vol) using only past data (shift(1)).
    Prevents look-ahead. Scale factor clipped to [0.25, 2.0].
    """
    daily_vol  = market_ret.rolling(current_window).std().shift(1)
    target_vol = market_ret.rolling(target_window).std().shift(1)
    target_vol = target_vol.ffill().bfill()
    daily_vol  = daily_vol.ffill().bfill()

    scale = (target_vol / daily_vol.replace(0, np.nan)).fillna(1.0).clip(0.25, 2.0)
    return (weight_series * scale).clip(0.0, max_cap)


def run_adaptive_allocator(train_df:        pd.DataFrame,
                            test_df:         pd.DataFrame,
                            hmm_lm:          dict,
                            hmm_prob_cols:   list,
                            label_col:       str   = "Resolved_Label",
                            return_col:      str   = "Market_Return",
                            window_label:    str   = "") -> pd.Series:
    """
    Full adaptive allocation pipeline for one walk-forward window.
    All parameters estimated on train_df only. Applied to test_df.

    Steps
    -----
    1. estimate_regime_forward_stats on train
    2. compute_regime_weights_from_stats → base weight per emotion
    3. For each test day: probability-weighted blend of emotion weights
       using HMM posterior probabilities
    4. Fallback to active Resolved_Label weight if probs unavailable
    5. Volatility targeting
    6. Clip to [0, MAX_WEIGHT]

    Returns
    -------
    pd.Series: daily final allocation weight, index = test_df.index
    Saves forward stats to _regime_fwd_stats_rows for CSV export.
    """
    # Step 1: forward stats on train
    stats_df = estimate_regime_forward_stats(train_df, label_col, return_col)

    # Store for CSV
    for regime, row in stats_df.iterrows():
        _regime_fwd_stats_rows.append({
            "window_start"       : window_label,
            "regime"             : regime,
            "avg_forward_return" : round(row["avg_forward_return"], 8),
            "forward_volatility" : round(row["forward_volatility"],  8),
            "sharpe_like_score"  : round(row["sharpe_like_score"],   6),
            "observations"       : int(row["observations"]),
        })

    # Step 2: base weights from forward stats
    base_weights = compute_regime_weights_from_stats(stats_df,
                                                      min_weight=0.10,
                                                      max_weight=1.00)

    # Step 3: probability-weighted blend on test
    hmm_prob_df_test = test_df[[c for c in hmm_prob_cols if c in test_df.columns]]
    # Build cluster_int → emotion map (hmm_lm maps cluster_int → emotion)
    cluster_to_label = {int(k): v for k, v in hmm_lm.items()}

    if not hmm_prob_df_test.empty:
        prob_weight = compute_probability_weighted_exposure(
            hmm_prob_df_test, cluster_to_label, base_weights
        )
    else:
        # Fallback: hard assignment from Resolved_Label
        prob_weight = test_df[label_col].shift(1).map(base_weights).fillna(0.5)

    # Step 4: lag-1 (no look-ahead already handled in prob_weight via shift in fallback;
    # prob columns are model posteriors on today's features — use shift(1) for signal)
    prob_weight = prob_weight.shift(1).fillna(prob_weight.iloc[0]
                                               if len(prob_weight) > 0 else 0.5)

    # Step 5: volatility targeting
    vol_targeted = apply_volatility_targeting(prob_weight, test_df[return_col])

    # Step 6: clip
    final = vol_targeted.clip(0.0, MAX_WEIGHT)

    # Store per-day adaptive allocations for CSV
    for date in test_df.index:
        detected = test_df.loc[date, label_col] if label_col in test_df.columns else ""
        p_w      = float(prob_weight.loc[date])  if date in prob_weight.index  else np.nan
        b_w_val  = float(base_weights.get(str(detected), np.nan))
        v_w      = float(vol_targeted.loc[date]) if date in vol_targeted.index else np.nan
        f_w      = float(final.loc[date])        if date in final.index        else np.nan
        _adaptive_alloc_rows.append({
            "date"              : str(date.date()),
            "detected_regime"   : detected,
            "probability_weight": round(p_w,   6) if not np.isnan(p_w)   else "",
            "base_weight"       : round(b_w_val,6) if not np.isnan(b_w_val) else "",
            "vol_scaled_weight" : round(v_w,   6) if not np.isnan(v_w)   else "",
            "final_weight"      : round(f_w,   6) if not np.isnan(f_w)   else "",
        })

    return final

# ─────────────────────────────────────────────────────────────────────────────
# RISK MANAGEMENT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def apply_drawdown_stop(weight_series: pd.Series,
                         market_ret: pd.Series,
                         dd_stop: float = DD_STOP_THRESHOLD,
                         dd_resume: float = DD_RESUME_THRESHOLD,
                         max_cap: float = MAX_WEIGHT) -> tuple:
    """
    Circuit breaker: forces weight to 0 when cumulative drawdown from peak
    breaches dd_stop. Resumes trading when drawdown recovers above dd_resume.

    No look-ahead: decision on day T uses equity curve up to T-1.

    Returns
    -------
    adjusted_weights : pd.Series
    cash_flag        : pd.Series  (1 = in cash mode, 0 = trading)
    drawdowns        : pd.Series
    """
    weights_out = weight_series.copy().clip(0.0, max_cap)
    cash_flag   = pd.Series(0, index=weight_series.index)
    in_cash     = False
    equity      = 1.0
    peak        = 1.0
    drawdowns   = []

    for i, (date, w) in enumerate(weights_out.items()):
        # Compute drawdown BEFORE today's decision (lag-1 safety)
        dd = (equity - peak) / peak if peak > 0 else 0.0
        drawdowns.append(dd)

        if in_cash:
            if dd > dd_resume:       # recovered enough
                in_cash = False
            else:
                weights_out.iloc[i] = 0.0
                cash_flag.iloc[i]   = 1
        else:
            if dd < dd_stop:         # breached stop
                in_cash = True
                weights_out.iloc[i] = 0.0
                cash_flag.iloc[i]   = 1

        # Update equity with today's actual return (after decision)
        if date in market_ret.index:
            r = market_ret.loc[date]
            equity *= (1 + weights_out.iloc[i] * (np.exp(r) - 1))
            peak = max(peak, equity)

    dd_series = pd.Series(drawdowns, index=weight_series.index)
    return weights_out, cash_flag, dd_series


def apply_volatility_scaling(weight_series: pd.Series,
                              market_ret: pd.Series,
                              current_window: int = VOL_SCALE_WINDOW,
                              target_window:  int = VOL_TARGET_WINDOW,
                              max_cap: float  = MAX_WEIGHT) -> pd.Series:
    """
    Scales base weights by (target_vol / current_vol).
    When current volatility is elevated, exposure shrinks proportionally.
    No look-ahead: both vol estimates use only past data (shift(1)).

    adjusted_weight = base_weight * (target_vol / current_vol)
    clipped to [0, max_cap]
    """
    daily_vol    = market_ret.rolling(current_window).std().shift(1)
    target_vol   = market_ret.rolling(target_window).std().shift(1)

    # Where we have no target vol estimate yet, use the first available value
    target_vol   = target_vol.ffill().bfill()
    daily_vol    = daily_vol.ffill().bfill()

    scale_factor = (target_vol / daily_vol.replace(0, np.nan)).fillna(1.0)
    scale_factor = scale_factor.clip(0.25, 2.0)   # prevent extreme scaling

    adjusted = (weight_series * scale_factor).clip(0.0, max_cap)
    return adjusted


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICAL SIGNIFICANCE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def run_bootstrap_significance_test(master_df: pd.DataFrame,
                                     n_iter: int = N_BOOTSTRAP,
                                     label_col: str = "Resolved_Label",
                                     weight_col: str = "Weight",
                                     return_col: str = "Market_Return") -> dict:
    """
    Shuffles regime labels n_iter times and recomputes Sharpe each time.
    p-value = fraction of random Sharpes >= actual Sharpe.

    Uses actual weights structure: maps shuffled labels to the same weight
    values that the real strategy used (from the Weight column directly).
    No retraining. Pure label permutation test.
    """
    actual_ret    = master_df["Strategy_Return"].dropna()
    actual_simple = np.exp(actual_ret) - 1
    actual_sharpe = (actual_simple.mean() / actual_simple.std()) * np.sqrt(252) \
                    if actual_simple.std() > 0 else 0.0

    weights_arr  = master_df[weight_col].dropna().values
    returns_arr  = master_df.loc[master_df[weight_col].notna(), return_col].values
    n            = min(len(weights_arr), len(returns_arr))
    weights_arr  = weights_arr[:n]
    returns_arr  = returns_arr[:n]

    random_sharpes = []
    rng = np.random.default_rng(42)

    print(f"  [bootstrap] Running {n_iter} permutations ...")
    for _ in range(n_iter):
        shuffled_w = rng.permutation(weights_arr)
        rand_ret   = shuffled_w * returns_arr
        rand_s     = np.exp(rand_ret) - 1
        std        = rand_s.std()
        sh         = (rand_s.mean() / std) * np.sqrt(252) if std > 0 else 0.0
        random_sharpes.append(sh)

    random_sharpes = np.array(random_sharpes)
    p_value = (random_sharpes >= actual_sharpe).mean()

    print(f"  [bootstrap] Actual Sharpe = {actual_sharpe:.4f}")
    print(f"  [bootstrap] Random Sharpe mean = {random_sharpes.mean():.4f}  "
          f"std = {random_sharpes.std():.4f}")
    print(f"  [bootstrap] p-value = {p_value:.4f}  "
          f"({'✓ Significant' if p_value < 0.05 else '✗ Not significant at 5%'})")

    return {
        "actual_sharpe" : actual_sharpe,
        "random_sharpes": random_sharpes,
        "p_value"       : p_value,
    }


def run_excess_return_ttest(master_df: pd.DataFrame) -> dict:
    """
    Tests whether daily excess returns (strategy - benchmark) are
    statistically different from zero using a one-sample t-test.
    """
    excess = (np.exp(master_df["Strategy_Return"])
              - np.exp(master_df["Benchmark_Return"])).dropna()

    t_stat, p_val = _ttest_1samp(excess, popmean=0.0)

    ann_excess = excess.mean() * 252
    print(f"  [t-test] Annualised excess return = {ann_excess*100:+.3f}%")
    print(f"  [t-test] t-statistic = {t_stat:.4f}   p-value = {p_val:.4f}  "
          f"({'✓ Significant' if p_val < 0.05 else '✗ Not significant at 5%'})")

    return {"t_stat": t_stat, "p_value": p_val,
            "ann_excess_return": ann_excess}


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def calculate_regime_persistence(master_df: pd.DataFrame,
                                  label_col: str = "Resolved_Label") -> pd.DataFrame:
    """
    Measures how long the strategy stays in each regime without switching.
    Healthy macro regimes persist 20–60 days on average.
    2–3 day average signals noisy label flipping.
    """
    regimes = master_df[label_col].dropna()
    runs = []
    prev, cnt = None, 0
    for r in regimes:
        if r == prev:
            cnt += 1
        else:
            if prev is not None:
                runs.append({"Regime": prev, "Days": cnt})
            prev, cnt = r, 1
    if prev is not None:
        runs.append({"Regime": prev, "Days": cnt})

    if not runs:
        return pd.DataFrame()

    run_df = pd.DataFrame(runs)
    persist = (run_df.groupby("Regime")["Days"]
               .agg(Spells="count",
                    Avg_Days="mean",
                    Median_Days="median",
                    Max_Days="max",
                    Total_Days="sum")
               .round(1)
               .reset_index())

    print("\n  [persistence] Regime duration statistics (OOS):")
    for _, row in persist.iterrows():
        avg = row["Avg_Days"]
        flag = ("✓ healthy" if 20 <= avg <= 200
                else "⚠ too noisy" if avg < 20
                else "⚠ too sticky")
        print(f"    {row['Regime']:18s}  avg={avg:5.1f}d  "
              f"median={row['Median_Days']:5.1f}d  spells={int(row['Spells']):3d}  {flag}")

    return persist


def run_oos_validation_suite(master_df: pd.DataFrame) -> None:
    """
    Runs all validation checks on the stitched OOS master dataframe only.
    No training data is involved.
    """
    from sklearn.metrics import silhouette_score as _sil

    print("\n" + "=" * 60)
    print("PHASE 6c — OOS Validation Suite")
    print("=" * 60)

# Use all available enhanced features — select only cols present in both windows
    feat_cols_val = [c for c in FEAT_COLS_CORE if c in master_df.columns]

    # ── 1. Cluster quality on OOS data ──────────────────────────────────────
    print("\n[1] Cluster Quality (OOS HMM labels)")
    if "HMM_Regime" in master_df.columns and len(feat_cols_val) >= 2:
        try:
            X_val = StandardScaler().fit_transform(
                master_df[feat_cols_val].dropna()
            )
            labels_val = master_df["HMM_Regime"].loc[
                master_df[feat_cols_val].dropna().index
            ]
            if labels_val.nunique() >= 2:
                sil = _sil(X_val, labels_val,
                            sample_size=min(2000, len(X_val)), random_state=42)
                ch  = _ch_score(X_val, labels_val)
                print(f"  Silhouette Score    : {sil:.4f}  "
                      f"({'strong' if sil > 0.5 else 'moderate' if sil > 0.25 else 'weak'})")
                print(f"  Calinski-Harabasz   : {ch:.1f}  (higher = better separated)")
            else:
                print("  Insufficient unique labels for cluster quality metrics.")
        except Exception as e:
            print(f"  Cluster quality skipped: {e}")
    else:
        print("  HMM_Regime or features not available in OOS master.")

    # ── 2. Known event detection ─────────────────────────────────────────────
    print("\n[2] Known Event Detection (OOS data only)")
    known_events = [
        ("COVID Crash",      "2020-02-20", "2020-03-23",
         ["Despair", "Capitulation", "Pessimism"]),
        ("Vaccine Rally",    "2020-11-09", "2021-02-15",
         ["Hope", "Optimism", "Enthusiasm"]),
        ("2022 Bear Market", "2022-01-01", "2022-06-30",
         ["Euphoria", "Unease", "Denial", "Pessimism"]),
        ("2023 Recovery",    "2023-01-01", "2023-06-30",
         ["Hope", "Optimism"]),
        ("2024 Rally",       "2024-01-01", "2024-12-31",
         ["Enthusiasm", "Exhilaration"]),
    ]
    scores = []
    for name, s, e, expected in known_events:
        mask = (master_df.index >= s) & (master_df.index <= e)
        if mask.sum() == 0:
            print(f"  – {name}: not in OOS window")
            continue
        actual  = master_df.loc[mask, "Resolved_Label"].unique()
        matches = [r for r in actual if r in expected]
        pct     = len(matches) / len(expected) * 100
        scores.append(pct)
        icon = "✓" if pct >= 50 else "✗"
        print(f"  {icon} {name}: matched {matches} / expected {expected}  "
              f"({pct:.0f}%)")
    if scores:
        print(f"  Average detection accuracy: {np.mean(scores):.1f}%")

    # ── 3. Regime characteristic checks ─────────────────────────────────────
    print("\n[3] Regime Characteristics (OOS)")
    rstat = master_df.groupby("Resolved_Label")[
        ["Market_Return", "Volatility"]].mean()
    checks = 0
    total  = 0

    hr = rstat["Market_Return"].idxmax()
    total += 1
    if hr in ["Enthusiasm", "Exhilaration", "Optimism", "Euphoria"]:
        print(f"  ✓ Highest return regime: {hr}")
        checks += 1
    else:
        print(f"  ✗ Highest return regime: {hr}  (expected Bull state)")

    lr = rstat["Market_Return"].idxmin()
    total += 1
    if lr in ["Despair", "Capitulation", "Pessimism"]:
        print(f"  ✓ Lowest return regime : {lr}")
        checks += 1
    else:
        print(f"  ✗ Lowest return regime : {lr}  (expected Bear state)")

    hv = rstat["Volatility"].idxmax()
    total += 1
    if hv in ["Despair", "Capitulation", "Pessimism", "Unease", "Denial"]:
        print(f"  ✓ Highest vol regime   : {hv}")
        checks += 1
    else:
        print(f"  ✗ Highest vol regime   : {hv}  (expected Bear/Turning state)")

    print(f"  Passed {checks}/{total} characteristic checks.")

    # ── 4. Regime persistence ────────────────────────────────────────────────
    print("\n[4] Regime Persistence (OOS)")
    persist_df = calculate_regime_persistence(master_df)
    if not persist_df.empty:
        persist_df.to_csv("regime_persistence.csv", index=False)
        print("  Saved: regime_persistence.csv")

        # regime_duration_histogram.png
        fig_rh, ax_rh = plt.subplots(figsize=(12, 5))
        for _, row in persist_df.iterrows():
            regime = row["Regime"]
            color  = REGIME_COLORS.get(regime, "#888")
            ax_rh.bar(regime, row["Avg_Days"], color=color, alpha=0.85,
                      edgecolor="white")
            ax_rh.text(regime, row["Avg_Days"] + 0.3,
                       f"{row['Avg_Days']:.0f}d", ha="center",
                       fontsize=8, fontweight="bold")
        ax_rh.axhline(20,  color="#e74c3c", linestyle="--", linewidth=1,
                      label="Min healthy (20d)")
        ax_rh.axhline(200, color="#2980b9", linestyle="--", linewidth=1,
                      label="Max healthy (200d)")
        ax_rh.set_ylabel("Avg Spell Duration (days)")
        ax_rh.set_title("Regime Persistence — OOS Periods",
                         fontweight="bold", fontsize=13)
        ax_rh.set_xticklabels(persist_df["Regime"], rotation=40,
                               ha="right", fontsize=9)
        ax_rh.legend(fontsize=9); ax_rh.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig("regime_duration_histogram.png", dpi=150)
        plt.show()
        print("  Saved: regime_duration_histogram.png")

def run_single_window(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit all models on train_df, predict on test_df.
    Returns an augmented copy of test_df with regime labels, probabilities,
    weights, strategy/benchmark returns — NO data from the test period
    was used during model fitting.
    """
    feat_cols = ["Market_Return", "Volatility", "Dispersion",
                 "Avg_Correlation", "Breadth", "Momentum", "RSI"]

    X_train = train_df[feat_cols].values
    X_test  = test_df[feat_cols].values

    # ── scaler: fit on train only ──
    sc = StandardScaler()
    X_train_s = sc.fit_transform(X_train)
    X_test_s  = sc.transform(X_test)           # no fit here

    # ── KMeans ──
   # ── Step 0: select optimal k on THIS training window only ──────────────
    best_k, bic_win, sil_win = select_optimal_regime_count(X_train_s)
    # Store for diagnostics (returned via augmented test_df metadata)
    test_df = test_df.copy()
    test_df["_best_k"]       = best_k
    test_df["_bic_at_best_k"]  = bic_win[best_k]
    test_df["_sil_at_best_k"]  = sil_win.get(best_k, np.nan)

    # ── KMeans ──
    km = KMeans(n_clusters=best_k, random_state=42, n_init=30)
    km.fit(X_train_s)
    train_df = train_df.copy()
    train_df["KMeans_Regime"] = km.predict(X_train_s)
    test_df["KMeans_Regime"]  = km.predict(X_test_s)

    # ── GMM ──
    gm = GaussianMixture(n_components=best_k, covariance_type="full",
                          random_state=42, n_init=10, max_iter=300)
    gm.fit(X_train_s)
    train_df["GMM_Regime"] = gm.predict(X_train_s)
    test_df["GMM_Regime"]  = gm.predict(X_test_s)

    gmm_probs_test   = gm.predict_proba(X_test_s)
    gmm_prob_test_df = pd.DataFrame(
        gmm_probs_test, index=test_df.index,
        columns=[f"GMM_P_R{i}" for i in range(best_k)]
    )

    # ── HMM (or GMM proxy) ──
    if HMM_AVAILABLE:
        hm = GaussianHMM(n_components=best_k, covariance_type="full",
                          n_iter=300, random_state=42)
        hm.fit(X_train_s)
        train_df["HMM_Regime"] = hm.predict(X_train_s)
        test_df["HMM_Regime"]  = hm.predict(X_test_s)
        hmm_probs_test         = hm.predict_proba(X_test_s)
    else:
        train_df["HMM_Regime"] = train_df["GMM_Regime"]
        test_df["HMM_Regime"]  = test_df["GMM_Regime"]
        hmm_probs_test         = gmm_probs_test

    hmm_prob_test_df = pd.DataFrame(
        hmm_probs_test, index=test_df.index,
        columns=[f"HMM_P_R{i}" for i in range(best_k)]
    )
    
    # ── HSMM (or HMM proxy) ──
    if HSMM_AVAILABLE:
        hsm = GaussianHSMM(n_components=best_k, covariance_type="full",
                            n_iter=300, min_duration=MIN_REGIME_DURATION,
                            random_state=42)
        hsm.fit(X_train_s)
        train_df["HSMM_Regime"] = hsm.predict(X_train_s)
        test_df["HSMM_Regime"]  = hsm.predict(X_test_s)
        hsmm_probs_test         = hsm.predict_proba(X_test_s)
    else:
        train_df["HSMM_Regime"] = train_df["HMM_Regime"]
        test_df["HSMM_Regime"]  = test_df["HMM_Regime"]
        hsmm_probs_test         = hmm_probs_test

    hsmm_prob_test_df = pd.DataFrame(
        hsmm_probs_test, index=test_df.index,
        columns=[f"HSMM_P_R{i}" for i in range(best_k)]
    )

# Derive label maps from training data only — no fingerprints, pure cluster stats
    _wl = str(train_df.index[0].date())
    kmeans_lm = label_emotional_cycle(train_df, "KMeans_Regime", window_label=_wl)
    gmm_lm    = label_emotional_cycle(train_df, "GMM_Regime",    window_label=_wl)
    hmm_lm    = label_emotional_cycle(train_df, "HMM_Regime",    window_label=_wl)
    hsmm_lm   = label_emotional_cycle(train_df, "HSMM_Regime",   window_label=_wl)

    # Apply maps to test
    test_df["KMeans_Label"] = test_df["KMeans_Regime"].map(kmeans_lm)
    test_df["GMM_Label"]    = test_df["GMM_Regime"].map(gmm_lm)
    test_df["HMM_Label"]    = test_df["HMM_Regime"].map(hmm_lm)
    test_df["HSMM_Label"]   = test_df["HSMM_Regime"].map(hsmm_lm)

    # Attach probabilities
    test_df = pd.concat([test_df, gmm_prob_test_df, hmm_prob_test_df,
                         hsmm_prob_test_df], axis=1)

# ── resolve_regime needs a few rows of context before the test window ──
    # We take the last RESOLVE_WINDOW rows from train as a "warm-up" boundary guard
    boundary = train_df.tail(RESOLVE_WINDOW).copy()
    boundary["KMeans_Label"] = boundary["KMeans_Regime"].map(kmeans_lm)
    boundary["GMM_Label"]    = boundary["GMM_Regime"].map(gmm_lm)
    boundary["HMM_Label"]    = boundary["HMM_Regime"].map(hmm_lm)
    boundary["HSMM_Label"]   = boundary["HSMM_Regime"].map(hsmm_lm)
    boundary = pd.concat(
        [boundary,
         pd.DataFrame(index=boundary.index,
                      columns=[c for c in hmm_prob_test_df.columns],
                      data=gm.predict_proba(sc.transform(boundary[feat_cols].values)))],
        axis=1
    )

    combined = pd.concat([boundary, test_df])
    combined = combined[~combined.index.duplicated(keep="last")]

    # resolve_regime is the authority for transition logic
    resolved_combined = resolve_regime(combined, label_col="HMM_Label",
                                       return_col="Market_Return",
                                       window=RESOLVE_WINDOW)
    resolved_combined = resolved_combined.ffill().bfill()

    # Slice back to test window only
    test_df["HMM_Label"]      = test_df["HMM_Label"].fillna(method="ffill").fillna(method="bfill")
    test_df["Resolved_Label"] = resolved_combined.loc[test_df.index]
    # ── Persistence filter: suppress noisy short-duration regime flips ───────
    test_df["Resolved_Label"] = enforce_min_duration(
        test_df["Resolved_Label"].ffill().bfill(), min_days=MIN_REGIME_DURATION
    )

    # ── Strategy returns (test period only) ──
# ── Step: resolve training labels for weight optimization ─────────────
    # We need Resolved_Label on the training set to optimize weights.
    # Build it the same way as the test boundary — resolve on train only.
# ── Resolve training labels (needed for forward stats estimation) ──────
    train_df["HMM_Label"]      = train_df["HMM_Regime"].map(hmm_lm)
    train_df["Resolved_Label"] = resolve_regime(
        train_df, label_col="HMM_Label",
        return_col="Market_Return", window=RESOLVE_WINDOW
    ).ffill().bfill()

    # ── Adaptive allocation — fully data-driven, no hand-picked values ────
    _wl_str      = str(train_df.index[0].date())
    hmm_prob_cols = [c for c in test_df.columns if c.startswith("HMM_P_R")]

    final_weight = run_adaptive_allocator(
        train_df       = train_df,
        test_df        = test_df,
        hmm_lm         = hmm_lm,
        hmm_prob_cols  = hmm_prob_cols,
        label_col      = "Resolved_Label",
        return_col     = "Market_Return",
        window_label   = _wl_str,
    )

    # ── Drawdown circuit breaker applied on top of adaptive weights ───────
    final_weight, cash_flag, drawdown_series = apply_drawdown_stop(
        final_weight, test_df["Market_Return"]
    )
    final_weight = final_weight.clip(0.0, MAX_WEIGHT)

    # ── Also keep optimizer path for CSV diagnostics (non-allocating) ─────
    opt_weights = optimize_regime_weights(
        train_df, label_col="Resolved_Label", return_col="Market_Return"
    )
    diag = diagnose_weight_optimizer(
        opt_weights, train_df, test_df,
        label_col="Resolved_Label", return_col="Market_Return"
    )
    _window_diag_rows.append(diag)
    for regime, w in opt_weights.items():
        _window_weight_rows.append({
            "window_start"    : _wl_str,
            "window_end"      : str(train_df.index[-1].date()),
            "regime"          : regime,
            "optimized_weight": round(w, 6),
        })

    # ── Sensitivity on optimizer weights (audit only, not applied) ────────
    sens_df = run_weight_sensitivity_analysis(
        opt_weights, train_df,
        label_col="Resolved_Label", return_col="Market_Return"
    )
    _window_sens_results.append(sens_df.assign(
        window_start=_wl_str,
        window_end=str(train_df.index[-1].date())
    ))

    # ── Strategy returns (test period only) ──────────────────────────────
    prev_weight = final_weight.shift(1).fillna(0.5)
    turnover    = (final_weight - prev_weight).abs()
    tc_drag     = turnover * TRANS_COST

    strat_ret  = final_weight * test_df["Market_Return"] - tc_drag
    bench_ret  = test_df["Market_Return"]

    test_df["Signal"]           = test_df["Resolved_Label"].shift(1)
    test_df["Base_Weight"]      = final_weight          # adaptive = the base now
    test_df["Vol_Scaled_Weight"]= final_weight          # vol targeting already inside
    test_df["Weight"]           = final_weight
    test_df["Cash_Mode"]        = cash_flag
    test_df["Drawdown"]         = drawdown_series
    test_df["Turnover"]         = turnover
    test_df["TC_Drag"]          = tc_drag
    test_df["Strategy_Return"]  = strat_ret
    test_df["Benchmark_Return"] = bench_ret
    # ── Save enhanced predictions for CSV export ─────────────────────────────
    _hmm_prob_cols = [c for c in test_df.columns if c.startswith("HMM_P_R")]
    save_enhanced_predictions(test_df, _hmm_prob_cols,
                               window_label=str(train_df.index[0].date()))

    return test_df


def run_walk_forward_backtest(features: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling walk-forward:
      - TRAIN_YEARS of history → fit all models
      - TEST_YEARS  of history → predict only
    Stitches test-period slices into one out-of-sample equity curve.
    """
    feat_cols = [c for c in FEAT_COLS_CORE if c in features.columns]

    start_date = features.index.min()
    end_date   = features.index.max()

    # Build list of (train_start, train_end, test_start, test_end) windows
    windows = []
    train_start = start_date
    while True:
        train_end  = train_start + pd.DateOffset(years=TRAIN_YEARS)
        test_start = train_end
        test_end   = test_start + pd.DateOffset(years=TEST_YEARS)
        if test_end > end_date:
            # Use whatever is left as the final test window
            test_end = end_date
        if test_start >= end_date:
            break
        windows.append((train_start, train_end, test_start, test_end))
        train_start = train_start + pd.DateOffset(years=TEST_YEARS)  # roll forward

    print(f"  Walk-forward windows: {len(windows)}")
    for i, (ts, te, ss, se) in enumerate(windows):
        print(f"    [{i+1}] Train: {ts.date()} → {te.date()}  |  Test: {ss.date()} → {se.date()}")

    pieces = []
    for i, (train_s, train_e, test_s, test_e) in enumerate(windows):
        train_mask = (features.index >= train_s) & (features.index < train_e)
        test_mask  = (features.index >= test_s)  & (features.index <= test_e)

        train_df = features.loc[train_mask, feat_cols].copy()
        test_df  = features.loc[test_mask,  feat_cols].copy()

        if len(train_df) < 100 or len(test_df) < 5:
            print(f"    [window {i+1}] Skipped — insufficient data")
            continue

        print(f"\n  ── Window {i+1}/{len(windows)} ──")
        result = run_single_window(train_df, test_df)
        pieces.append(result)

    if not pieces:
        raise RuntimeError("Walk-forward produced no valid windows. Check date range.")

    master_wf = pd.concat(pieces).sort_index()
    master_wf = master_wf[~master_wf.index.duplicated(keep="last")]

    # Cumulative curves on stitched out-of-sample data
    master_wf["Cum_Strategy"]  = np.exp(master_wf["Strategy_Return"].cumsum())
    master_wf["Cum_Benchmark"] = np.exp(master_wf["Benchmark_Return"].cumsum())

    # ── Regime count diagnostics CSV ────────────────────────────────────────
    diag_rows = []
    for i, (train_s, train_e, test_s, test_e) in enumerate(windows):
        test_mask = (master_wf.index >= test_s) & (master_wf.index <= test_e)
        slice_df  = master_wf.loc[test_mask]
        if slice_df.empty:
            continue
        diag_rows.append({
            "window"     : i + 1,
            "train_start": str(train_s.date()),
            "train_end"  : str(train_e.date()),
            "test_start" : str(test_s.date()),
            "test_end"   : str(test_e.date()),
            "selected_k" : int(slice_df["_best_k"].iloc[0])
                           if "_best_k" in slice_df.columns else np.nan,
            "bic"        : round(slice_df["_bic_at_best_k"].iloc[0], 2)
                           if "_bic_at_best_k" in slice_df.columns else np.nan,
            "silhouette" : round(slice_df["_sil_at_best_k"].iloc[0], 6)
                           if "_sil_at_best_k" in slice_df.columns else np.nan,
        })

    if diag_rows:
        diag_df = pd.DataFrame(diag_rows)
        diag_df.to_csv("regime_count_diagnostics.csv", index=False)
        print("\n  ── Walk-Forward Regime Count Diagnostics ──")
        print(diag_df.to_string(index=False))
        print("  Saved: regime_count_diagnostics.csv")

        # Per-window BIC / silhouette bar chart
        fig_wd, (ax_w1, ax_w2) = plt.subplots(1, 2, figsize=(14, 5))
        w_labels = [f"W{r['window']}\n{r['train_start'][:4]}→{r['test_start'][:4]}"
                    for r in diag_rows]

        ax_w1.bar(w_labels, diag_df["selected_k"], color="#2980b9", alpha=0.85)
        ax_w1.set_ylabel("Selected k"); ax_w1.set_title("Optimal k per Walk-Forward Window",
                                                          fontweight="bold")
        ax_w1.grid(alpha=0.3, axis="y")
        for xi, v in enumerate(diag_df["selected_k"]):
            ax_w1.text(xi, v + 0.1, str(int(v)), ha="center", fontsize=10, fontweight="bold")

        ax_w2.bar(w_labels, diag_df["silhouette"], color="#27ae60", alpha=0.85)
        ax_w2.set_ylabel("Silhouette Score")
        ax_w2.set_title("Silhouette at Selected k per Window", fontweight="bold")
        ax_w2.grid(alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig("bic_vs_clusters.png",        dpi=150)
        plt.savefig("silhouette_vs_clusters.png", dpi=150)
        plt.show()
        print("  Saved: bic_vs_clusters.png  |  silhouette_vs_clusters.png  (per-window view)")

    # Drop internal metadata columns before returning
    meta_cols = [c for c in master_wf.columns if c.startswith("_")]
    master_wf = master_wf.drop(columns=meta_cols, errors="ignore")

    return master_wf

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6a — WALK-FORWARD BACKTEST  (true out-of-sample)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 6 — Walk-Forward Backtest  (out-of-sample only)")
print(f"  Config: train={TRAIN_YEARS}yr  test={TEST_YEARS}yr  "
      f"tx_cost={TRANS_COST*10000:.0f}bps  resolve_window={RESOLVE_WINDOW}")
print("=" * 60)

# Allocation weights based on emotional state


# Run the walk-forward engine
master = run_walk_forward_backtest(features)

# ── Save optimized_regime_weights.csv ────────────────────────────────────────
if _window_weight_rows:
    weights_df = pd.DataFrame(_window_weight_rows)
    weights_df.to_csv("optimized_regime_weights.csv", index=False)
    print("  Saved: optimized_regime_weights.csv")

    # ── optimized_weights_heatmap.png ────────────────────────────────────────
    try:
        _pivot = weights_df.pivot(index="window_start", columns="regime",
                                   values="optimized_weight")
        fig_wh, ax_wh = plt.subplots(figsize=(max(8, len(_pivot.columns) * 1.2),
                                               max(4, len(_pivot) * 0.8)))
        im_wh = ax_wh.imshow(_pivot.values, cmap="RdYlGn", vmin=0, vmax=1,
                              aspect="auto")
        plt.colorbar(im_wh, ax=ax_wh, label="Allocation Weight")
        ax_wh.set_xticks(range(len(_pivot.columns)))
        ax_wh.set_xticklabels(_pivot.columns, rotation=40, ha="right", fontsize=9)
        ax_wh.set_yticks(range(len(_pivot.index)))
        ax_wh.set_yticklabels(_pivot.index, fontsize=8)
        ax_wh.set_xlabel("Emotional Regime"); ax_wh.set_ylabel("Training Window Start")
        ax_wh.set_title("Optimized Allocation Weights per Walk-Forward Window",
                         fontweight="bold", fontsize=13)
        for i in range(len(_pivot.index)):
            for j in range(len(_pivot.columns)):
                v = _pivot.values[i, j]
                if not np.isnan(v):
                    ax_wh.text(j, i, f"{v:.2f}", ha="center", va="center",
                               fontsize=8, color="black" if 0.3 < v < 0.7 else "white")
        plt.tight_layout()
        plt.savefig("optimized_weights_heatmap.png", dpi=150)
        plt.show()
        print("  Saved: optimized_weights_heatmap.png")
    except Exception as _e:
        print(f"  [weights heatmap] Skipped: {_e}")

# ── Save weight_sensitivity_results.csv ──────────────────────────────────────
if _window_sens_results:
    all_sens_df = pd.concat(_window_sens_results, ignore_index=True)
    all_sens_df.to_csv("weight_sensitivity_results.csv", index=False)
    print("  Saved: weight_sensitivity_results.csv")

    # ── weight_sensitivity_surface.png ───────────────────────────────────────
    try:
        _avg_sens = (all_sens_df.groupby(["regime", "perturbation"])["sharpe"]
                     .mean().reset_index())
        _regimes_s  = sorted(_avg_sens["regime"].unique())
        _perts      = sorted(_avg_sens["perturbation"].unique())
        _mat        = np.full((len(_regimes_s), len(_perts)), np.nan)
        for i, r in enumerate(_regimes_s):
            for j, p in enumerate(_perts):
                v = _avg_sens.loc[(_avg_sens["regime"] == r) &
                                   (_avg_sens["perturbation"] == p), "sharpe"]
                if not v.empty:
                    _mat[i, j] = v.values[0]

        fig_ss, ax_ss = plt.subplots(figsize=(6, max(4, len(_regimes_s) * 0.7)))
        im_ss = ax_ss.imshow(_mat, cmap="RdYlGn", aspect="auto")
        plt.colorbar(im_ss, ax=ax_ss, label="Avg Sharpe")
        ax_ss.set_xticks(range(len(_perts)))
        ax_ss.set_xticklabels(_perts, fontsize=9)
        ax_ss.set_yticks(range(len(_regimes_s)))
        ax_ss.set_yticklabels(_regimes_s, fontsize=9)
        ax_ss.set_xlabel("Perturbation"); ax_ss.set_ylabel("Regime")
        ax_ss.set_title("Weight Sensitivity — Avg Sharpe Across Windows",
                         fontweight="bold", fontsize=12)
        for i in range(len(_regimes_s)):
            for j in range(len(_perts)):
                if not np.isnan(_mat[i, j]):
                    ax_ss.text(j, i, f"{_mat[i,j]:.2f}", ha="center", va="center",
                               fontsize=8)
        plt.tight_layout()
        plt.savefig("weight_sensitivity_surface.png", dpi=150)
        plt.show()
        print("  Saved: weight_sensitivity_surface.png")
    except Exception as _e:
        print(f"  [sensitivity surface] Skipped: {_e}")
        # ── Save weight_optimizer_diagnostics.csv ────────────────────────────────────
if _window_diag_rows:
    diag_out = pd.DataFrame(_window_diag_rows)
    diag_out.to_csv("weight_optimizer_diagnostics.csv", index=False)
    print("  Saved: weight_optimizer_diagnostics.csv")
    print(diag_out.to_string(index=False))

    # ── train_vs_test_optimizer_performance.png ───────────────────────────────
    try:
        fig_tt, axes_tt = plt.subplots(1, 3, figsize=(16, 5))
        w_labels = [f"W{i+1}\n{r['window_start'][:7]}"
                    for i, r in enumerate(_window_diag_rows)]

        for ax, col, title, color in [
            (axes_tt[0], "train_sharpe",  "Sharpe Ratio",  "#2980b9"),
            (axes_tt[1], "train_cagr",    "CAGR",          "#27ae60"),
            (axes_tt[2], "avg_exposure",  "Avg Exposure",  "#8e44ad"),
        ]:
            train_vals = pd.to_numeric(diag_out[col], errors="coerce").tolist()
            test_col   = col.replace("train_", "test_")
            test_vals  = (pd.to_numeric(diag_out[test_col], errors="coerce").tolist()
                          if test_col in diag_out.columns else [])

            x = range(len(w_labels))
            ax.bar(x, train_vals, alpha=0.6, color=color, label="Train")
            if test_vals:
                ax.bar(x, test_vals, alpha=0.0, color=color,
                       edgecolor="black", linewidth=1.5,
                       fill=False, label="OOS Test")
            ax.set_xticks(list(x))
            ax.set_xticklabels(w_labels, fontsize=8)
            ax.set_title(title, fontweight="bold")
            ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

            if col == "avg_exposure":
                ax.axhline(OPT_MIN_EXPOSURE, color="#e74c3c", linestyle="--",
                           linewidth=1.2,
                           label=f"Min floor ({OPT_MIN_EXPOSURE:.0%})")
                ax.legend(fontsize=8)

        plt.suptitle("Walk-Forward Optimizer: Train vs OOS Performance",
                     fontweight="bold", fontsize=13)
        plt.tight_layout()
        plt.savefig("train_vs_test_optimizer_performance.png", dpi=150)
        plt.show()
        print("  Saved: train_vs_test_optimizer_performance.png")
    except Exception as _e:
        print(f"  [optimizer chart] Skipped: {_e}")

# ── Save regime_forward_stats.csv ─────────────────────────────────────────────
if _regime_fwd_stats_rows:
    fwd_df = pd.DataFrame(_regime_fwd_stats_rows)
    fwd_df.to_csv("regime_forward_stats.csv", index=False)
    print("  Saved: regime_forward_stats.csv")

    try:
        avg_fwd = (fwd_df.groupby("regime")["avg_forward_return"].mean()
                   .sort_values())
        fig_re, ax_re = plt.subplots(figsize=(12, 5))
        colors_re = [REGIME_COLORS.get(r, "#888") for r in avg_fwd.index]
        ax_re.bar(avg_fwd.index, avg_fwd.values * 100,
                  color=colors_re, alpha=0.85, edgecolor="white")
        ax_re.axhline(0, color="grey", linewidth=0.8, linestyle=":")
        ax_re.set_ylabel("Avg Forward Daily Return (%)")
        ax_re.set_title("Regime Expected Forward Returns (averaged across windows)",
                         fontweight="bold", fontsize=13)
        ax_re.set_xticklabels(avg_fwd.index, rotation=40, ha="right", fontsize=9)
        ax_re.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig("regime_expected_returns.png", dpi=150)
        plt.show()
        print("  Saved: regime_expected_returns.png")
    except Exception as _e:
        print(f"  [regime_expected_returns] Skipped: {_e}")

# ── Save adaptive_allocations.csv ─────────────────────────────────────────────
if _adaptive_alloc_rows:
    alloc_df = pd.DataFrame(_adaptive_alloc_rows)
    alloc_df.to_csv("adaptive_allocations.csv", index=False)
    print("  Saved: adaptive_allocations.csv")
    # ── Save enhanced_regime_predictions.csv ─────────────────────────────────────
if _enhanced_pred_rows:
    enh_df = pd.DataFrame(_enhanced_pred_rows)
    enh_df.to_csv("enhanced_regime_predictions.csv", index=False)
    print("  Saved: enhanced_regime_predictions.csv")

# ── Save feature_importance_regime.csv ───────────────────────────────────────
if "Resolved_Label" in master.columns:
    try:
        feat_cols_present = [c for c in FEAT_COLS_CORE if c in master.columns]
        var_by_state = (master.groupby("Resolved_Label")[feat_cols_present]
                        .var().mean(axis=0)
                        .sort_values(ascending=False))
        feat_imp_df = pd.DataFrame({
            "feature_name"    : var_by_state.index,
            "variance_by_state": var_by_state.values.round(8),
            "usefulness_rank" : range(1, len(var_by_state) + 1),
        })
        feat_imp_df.to_csv("feature_importance_regime.csv", index=False)
        print("  Saved: feature_importance_regime.csv")

        # feature_separation_heatmap.png
        _feat_means = (master.groupby("Resolved_Label")[feat_cols_present[:12]]
                       .mean())
        _feat_norm  = (_feat_means - _feat_means.mean()) / (_feat_means.std() + 1e-9)
        fig_fsh, ax_fsh = plt.subplots(
            figsize=(max(10, len(feat_cols_present[:12]) * 0.9),
                     max(5,  len(_feat_norm) * 0.6))
        )
        im_fsh = ax_fsh.imshow(_feat_norm.values, cmap="RdYlGn", aspect="auto")
        plt.colorbar(im_fsh, ax=ax_fsh, label="Normalised Mean")
        ax_fsh.set_xticks(range(len(_feat_norm.columns)))
        ax_fsh.set_xticklabels(_feat_norm.columns, rotation=45,
                                ha="right", fontsize=8)
        ax_fsh.set_yticks(range(len(_feat_norm.index)))
        ax_fsh.set_yticklabels(_feat_norm.index, fontsize=9)
        ax_fsh.set_title("Feature Mean by Regime — OOS Periods",
                          fontweight="bold", fontsize=13)
        plt.tight_layout()
        plt.savefig("feature_separation_heatmap.png", dpi=150)
        plt.show()
        print("  Saved: feature_separation_heatmap.png")
    except Exception as _e:
        print(f"  [feature heatmap] Skipped: {_e}")

    # state_probability_chart.png
    try:
        if _enhanced_pred_rows:
            ep = pd.DataFrame(_enhanced_pred_rows)
            ep["date"] = pd.to_datetime(ep["date"])
            ep = ep.set_index("date")
            ep["regime_probability"] = pd.to_numeric(
                ep["regime_probability"], errors="coerce"
            )
            ep["entropy"] = pd.to_numeric(ep["entropy"], errors="coerce")

            fig_sp, (ax_sp1, ax_sp2) = plt.subplots(
                2, 1, figsize=(18, 8), sharex=True
            )
            ax_sp1.fill_between(ep.index, ep["regime_probability"].fillna(0),
                                color="#2980b9", alpha=0.65,
                                label="Top regime probability")
            ax_sp1.axhline(0.6, color="#27ae60", linestyle="--",
                           linewidth=1, label="High confidence (0.6)")
            ax_sp1.axhline(0.4, color="#e74c3c", linestyle="--",
                           linewidth=1, label="Low confidence (0.4)")
            ax_sp1.set_ylabel("Probability"); ax_sp1.legend(fontsize=9)
            ax_sp1.set_title("State Probability & Entropy — OOS",
                              fontweight="bold", fontsize=13)
            ax_sp1.grid(alpha=0.3)

            ax_sp2.plot(ep.index, ep["entropy"].fillna(0),
                        color="#8e44ad", linewidth=1.2, label="Entropy")
            ax_sp2.set_ylabel("Entropy"); ax_sp2.set_xlabel("Date")
            ax_sp2.legend(fontsize=9); ax_sp2.grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig("state_probability_chart.png", dpi=150)
            plt.show()
            print("  Saved: state_probability_chart.png")
    except Exception as _e:
        print(f"  [state probability chart] Skipped: {_e}")

    # regime_timeline_enhanced.png
    try:
        fig_rte, (ax_rte1, ax_rte2) = plt.subplots(
            2, 1, figsize=(18, 9), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]}
        )
        ax_rte1.plot(master["Cum_Strategy"],  color="#27ae60",
                     linewidth=1.8, label="Strategy")
        ax_rte1.plot(master["Cum_Benchmark"], color="#e74c3c",
                     linewidth=1.8, linestyle="--", label="Benchmark")
        ax_rte1.set_ylabel("Portfolio Value")
        ax_rte1.set_title("Enhanced Regime Timeline — OOS",
                           fontweight="bold", fontsize=13)
        ax_rte1.legend(fontsize=9); ax_rte1.grid(alpha=0.3)

        for date, regime in master["Resolved_Label"].items():
            ax_rte2.axvline(date,
                            color=REGIME_COLORS.get(str(regime).split("_")[0], "#888"),
                            linewidth=0.5, alpha=0.7)
        ax_rte2.set_ylabel("Regime"); ax_rte2.set_yticks([])

        _patches_rte = [
            mpatches.Patch(color=c, label=r[:5], alpha=0.8)
            for r, c in REGIME_COLORS.items()
            if any(str(v).startswith(r) for v in master["Resolved_Label"].unique())
        ]
        ax_rte2.legend(handles=_patches_rte, loc="upper left",
                       fontsize=7, ncol=5)
        plt.tight_layout()
        plt.savefig("regime_timeline_enhanced.png", dpi=150)
        plt.show()
        print("  Saved: regime_timeline_enhanced.png")
    except Exception as _e:
        print(f"  [regime timeline enhanced] Skipped: {_e}")

    # average_state_duration.png — reuse persist_df from run_oos_validation_suite
    # (already saved as regime_duration_histogram.png in Phase 6c; alias it)
    try:
        import shutil
        shutil.copy("regime_duration_histogram.png", "average_state_duration.png")
        print("  Saved: average_state_duration.png (copy of regime_duration_histogram.png)")
    except Exception:
        pass

    try:
        alloc_df["date"] = pd.to_datetime(alloc_df["date"])
        alloc_df = alloc_df.set_index("date")

        fig_ae, (ax_ae1, ax_ae2) = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
        ax_ae1.plot(master["Cum_Strategy"],  color="#27ae60",
                    linewidth=1.8, label="Adaptive Strategy")
        ax_ae1.plot(master["Cum_Benchmark"], color="#e74c3c",
                    linewidth=1.8, linestyle="--", label="Buy & Hold")
        ax_ae1.set_ylabel("Portfolio Value"); ax_ae1.legend(fontsize=9)
        ax_ae1.set_title("Adaptive Allocation Strategy vs Benchmark",
                          fontweight="bold", fontsize=13)
        ax_ae1.grid(alpha=0.3)

        if "final_weight" in alloc_df.columns:
            fw = pd.to_numeric(alloc_df["final_weight"], errors="coerce").fillna(0)
            ax_ae2.fill_between(alloc_df.index, fw * 100,
                                color="#2980b9", alpha=0.6,
                                label="Final allocation %")
        ax_ae2.set_ylabel("Allocation (%)"); ax_ae2.set_xlabel("Date")
        ax_ae2.legend(fontsize=9); ax_ae2.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("adaptive_exposure_curve.png", dpi=150)
        plt.show()
        print("  Saved: adaptive_exposure_curve.png")
    except Exception as _e:
        print(f"  [adaptive_exposure_curve] Skipped: {_e}")

    try:
        fig_pw, ax_pw = plt.subplots(figsize=(18, 5))
        if "probability_weight" in alloc_df.columns:
            pw = pd.to_numeric(alloc_df["probability_weight"],
                               errors="coerce").ffill()
            ax_pw.plot(alloc_df.index, pw * 100, color="#8e44ad",
                       linewidth=1.2, label="Probability-weighted exposure")
        if "vol_scaled_weight" in alloc_df.columns:
            vw = pd.to_numeric(alloc_df["vol_scaled_weight"],
                               errors="coerce").ffill()
            ax_pw.plot(alloc_df.index, vw * 100, color="#27ae60",
                       linewidth=1.2, linestyle="--", label="After vol targeting")
        ax_pw.set_ylabel("Exposure (%)"); ax_pw.set_xlabel("Date")
        ax_pw.set_title("Probability-Weighted vs Vol-Targeted Allocations",
                         fontweight="bold", fontsize=13)
        ax_pw.legend(fontsize=9); ax_pw.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("probability_weighted_allocations.png", dpi=150)
        plt.show()
        print("  Saved: probability_weighted_allocations.png")
    except Exception as _e:
        print(f"  [probability_weighted] Skipped: {_e}")

# ── Metrics helper ──────────────────────────────────────────────────────────
def compute_metrics(log_ret_series, label):
    simple = np.exp(log_ret_series.dropna()) - 1
    cum    = np.exp(log_ret_series.dropna().sum()) - 1
    n_yrs  = len(simple) / 252
    cagr   = (1 + cum) ** (1 / max(n_yrs, 0.01)) - 1
    sharpe = (simple.mean() / simple.std()) * np.sqrt(252) if simple.std() > 0 else 0
    cw     = (1 + simple).cumprod()
    max_dd = ((cw - cw.cummax()) / cw.cummax()).min()
    win_r  = (simple > 0).mean()
    print(f"\n  {label}")
    print(f"  Period   : {log_ret_series.index[0].date()} → {log_ret_series.index[-1].date()}  "
          f"({n_yrs:.1f} yrs, out-of-sample only)")
    print(f"  CAGR     : {cagr*100:.2f}%")
    print(f"  Sharpe   : {sharpe:.3f}")
    print(f"  Max DD   : {max_dd*100:.2f}%")
    print(f"  Win Rate : {win_r*100:.1f}%")
    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": max_dd, "WinRate": win_r}

m_s = compute_metrics(master["Strategy_Return"],  "Emotion Cycle Strategy (OOS)")
m_b = compute_metrics(master["Benchmark_Return"], "Buy & Hold (OOS)")

alpha = m_s["CAGR"] - m_b["CAGR"]
print(f"\n  Alpha vs benchmark: {alpha*100:+.2f}% per year")

# ── Turnover / regime duration stats ────────────────────────────────────────
avg_turnover = master["Turnover"].mean()
print(f"\n  Avg daily turnover : {avg_turnover*100:.3f}%")
print(f"  Total TC drag      : {master['TC_Drag'].sum()*100:.2f}% cumulative")

regime_runs = []
prev_r, cnt = None, 0
for r in master["Resolved_Label"].dropna():
    if r == prev_r:
        cnt += 1
    else:
        if prev_r:
            regime_runs.append({"Regime": prev_r, "Days": cnt})
        prev_r, cnt = r, 1
if prev_r:
    regime_runs.append({"Regime": prev_r, "Days": cnt})

if regime_runs:
    dur_df = (pd.DataFrame(regime_runs)
              .groupby("Regime")["Days"]
              .agg(Spells="count", Avg="mean", Median="median", Max="max")
              .round(1))
    print("\n  Regime Duration Statistics (OOS periods):")
    print(dur_df.to_string())

# ── Probability confidence logging ──────────────────────────────────────────
hmm_prob_cols = [c for c in master.columns if c.startswith("HMM_P_R")]
if hmm_prob_cols:
    max_prob = master[hmm_prob_cols].max(axis=1)
    print(f"\n  Regime confidence (max HMM prob):")
    print(f"    Mean  : {max_prob.mean():.3f}")
    print(f"    Median: {max_prob.median():.3f}")
    print(f"    >0.6  : {(max_prob > 0.6).mean()*100:.1f}% of days")

# ── Equity curve chart ───────────────────────────────────────────────────────
cum_strategy  = master["Cum_Strategy"]
cum_benchmark = master["Cum_Benchmark"]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 9), sharex=True,
                                gridspec_kw={"height_ratios": [3, 1]})
ax1.plot(cum_strategy,  label="Emotion Cycle Strategy (OOS)", color="#27ae60", linewidth=2)
ax1.plot(cum_benchmark, label="Buy & Hold (OOS)",             color="#e74c3c", linewidth=2, linestyle="--")
ax1.set_ylabel("Portfolio Value")
ax1.legend(fontsize=10); ax1.grid(alpha=0.3)
ax1.set_title("Emotion Cycle Strategy vs Buy & Hold — Walk-Forward Out-of-Sample",
              fontsize=14, fontweight="bold")
ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:.2f}x"))

# Shade the out-of-sample windows on the x-axis bar
label_col_plot = "Resolved_Label" if "Resolved_Label" in master.columns else "HMM_Label"
for date, regime in master[label_col_plot].items():
    ax2.axvline(date, color=REGIME_COLORS.get(regime, "grey"),
                linewidth=0.6, alpha=0.7)
ax2.set_ylabel("Regime"); ax2.set_yticks([]); ax2.set_xlabel("Date")
patches = [mpatches.Patch(color=c, label=r[:4], alpha=0.8)
           for r, c in REGIME_COLORS.items()
           if r in master[label_col_plot].values]
ax2.legend(handles=patches, loc="upper left", fontsize=7, ncol=5)

plt.tight_layout()
plt.savefig("backtest_emotion_cycle.png", dpi=150)
plt.show()
print("  Saved: backtest_emotion_cycle.png")

# Save master (OOS only)
master.to_csv("regime_master_data.csv")
print("  Saved: regime_master_data.csv  (out-of-sample rows only)")
# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6c — OOS VALIDATION SUITE
# ─────────────────────────────────────────────────────────────────────────────
run_oos_validation_suite(master)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6d — STATISTICAL SIGNIFICANCE TESTING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 6d — Statistical Significance Testing")
print("=" * 60)

boot_results = run_bootstrap_significance_test(master)
ttest_results = run_excess_return_ttest(master)

# Save significance_test_results.csv
actual_simple = np.exp(master["Strategy_Return"].dropna()) - 1
actual_sharpe = boot_results["actual_sharpe"]

sig_rows = [
    {
        "metric"         : "Sharpe_Ratio",
        "actual_value"   : round(actual_sharpe, 4),
        "bootstrap_pvalue": round(boot_results["p_value"], 4),
        "t_stat"         : "",
        "p_value"        : "",
    },
    {
        "metric"         : "Excess_Return",
        "actual_value"   : round(ttest_results["ann_excess_return"] * 100, 4),
        "bootstrap_pvalue": "",
        "t_stat"         : round(ttest_results["t_stat"], 4),
        "p_value"        : round(ttest_results["p_value"], 4),
    },
]
sig_df = pd.DataFrame(sig_rows)
sig_df.to_csv("significance_test_results.csv", index=False)
print(f"\n  Saved: significance_test_results.csv")

# bootstrap_distribution.png
fig_bt, ax_bt = plt.subplots(figsize=(10, 5))
ax_bt.hist(boot_results["random_sharpes"], bins=50,
           color="#2980b9", alpha=0.75, edgecolor="white",
           label=f"Random Sharpes (n={N_BOOTSTRAP})")
ax_bt.axvline(actual_sharpe, color="#e74c3c", linewidth=2.5,
              label=f"Actual Sharpe = {actual_sharpe:.3f}")
ax_bt.axvline(np.percentile(boot_results["random_sharpes"], 95),
              color="#f39c12", linewidth=1.5, linestyle="--",
              label="95th pct of random")
ax_bt.set_xlabel("Sharpe Ratio"); ax_bt.set_ylabel("Frequency")
ax_bt.set_title(f"Bootstrap Significance Test  "
                f"(p = {boot_results['p_value']:.4f})",
                fontweight="bold", fontsize=13)
ax_bt.legend(fontsize=9); ax_bt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("bootstrap_distribution.png", dpi=150)
plt.show()
print("  Saved: bootstrap_distribution.png")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6e — RISK MANAGEMENT CHARTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 6e — Risk Management Diagnostics")
print("=" * 60)

# risk_management_log.csv
risk_cols = ["Base_Weight", "Weight", "Drawdown", "Cash_Mode"]
risk_cols_present = [c for c in risk_cols if c in master.columns]
if risk_cols_present:
    risk_log = master[risk_cols_present].copy()
    risk_log.index.name = "date"
    risk_log.columns = [c.lower().replace("weight", "adjusted_weight")
                        if c == "Weight" else
                        c.lower().replace("base_weight", "base_weight")
                        for c in risk_log.columns]
    # Rename for required CSV schema
    col_map = {
        "base_weight"     : "base_weight",
        "weight"          : "adjusted_weight",
        "drawdown"        : "drawdown",
        "cash_mode"       : "cash_mode_flag",
    }
    risk_log = master[risk_cols_present].rename(
        columns={"Weight": "adjusted_weight",
                 "Base_Weight": "base_weight",
                 "Drawdown": "drawdown",
                 "Cash_Mode": "cash_mode_flag"}
    )
    risk_log.index.name = "date"
    risk_log.to_csv("risk_management_log.csv")
    print("  Saved: risk_management_log.csv")

# drawdown_control_chart.png
if "Drawdown" in master.columns:
    fig_dc, (ax_dc1, ax_dc2, ax_dc3) = plt.subplots(
        3, 1, figsize=(18, 11), sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.5, 1]}
    )
    # Panel 1: equity curves
    ax_dc1.plot(master["Cum_Strategy"],  color="#27ae60", linewidth=1.8,
                label="Risk-Managed Strategy")
    ax_dc1.plot(master["Cum_Benchmark"], color="#e74c3c", linewidth=1.8,
                linestyle="--", label="Buy & Hold")
    if "Cash_Mode" in master.columns:
        cash_periods = master["Cash_Mode"] == 1
        ax_dc1.fill_between(master.index, ax_dc1.get_ylim()[0],
                            master["Cum_Strategy"],
                            where=cash_periods, alpha=0.15,
                            color="#e74c3c", label="Cash mode active")
    ax_dc1.set_ylabel("Portfolio Value"); ax_dc1.legend(fontsize=9)
    ax_dc1.set_title("Risk-Managed Strategy — Drawdown Control",
                     fontweight="bold", fontsize=13)
    ax_dc1.grid(alpha=0.3)

    # Panel 2: drawdown
    ax_dc2.fill_between(master.index, master["Drawdown"] * 100, 0,
                        color="#e74c3c", alpha=0.6, label="Drawdown %")
    ax_dc2.axhline(DD_STOP_THRESHOLD * 100, color="#c0392b",
                   linestyle="--", linewidth=1.2,
                   label=f"Stop threshold ({DD_STOP_THRESHOLD*100:.0f}%)")
    ax_dc2.axhline(DD_RESUME_THRESHOLD * 100, color="#f39c12",
                   linestyle="--", linewidth=1.2,
                   label=f"Resume threshold ({DD_RESUME_THRESHOLD*100:.0f}%)")
    ax_dc2.set_ylabel("Drawdown (%)"); ax_dc2.legend(fontsize=8)
    ax_dc2.grid(alpha=0.3)

    # Panel 3: weight over time
    ax_dc3.plot(master.index, master["Weight"] * 100,
                color="#2980b9", linewidth=1, label="Final weight %")
    if "Base_Weight" in master.columns:
        ax_dc3.plot(master.index, master["Base_Weight"] * 100,
                    color="#95a5a6", linewidth=0.8, linestyle="--",
                    label="Base weight %", alpha=0.7)
    ax_dc3.set_ylabel("Allocation (%)"); ax_dc3.set_xlabel("Date")
    ax_dc3.legend(fontsize=8); ax_dc3.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("drawdown_control_chart.png", dpi=150)
    plt.show()
    print("  Saved: drawdown_control_chart.png")

# volatility_scaled_weights.png
if "Vol_Scaled_Weight" in master.columns and "Base_Weight" in master.columns:
    fig_vs, (ax_vs1, ax_vs2) = plt.subplots(2, 1, figsize=(18, 8), sharex=True)

    ax_vs1.plot(master.index, master["Base_Weight"] * 100,
                color="#95a5a6", linewidth=1, label="Base weight (optimized)")
    ax_vs1.plot(master.index, master["Vol_Scaled_Weight"] * 100,
                color="#2980b9", linewidth=1.5, label="After vol scaling")
    ax_vs1.plot(master.index, master["Weight"] * 100,
                color="#27ae60", linewidth=1.5, label="Final (after DD stop)")
    ax_vs1.set_ylabel("Allocation (%)"); ax_vs1.legend(fontsize=9)
    ax_vs1.set_title("Volatility-Scaled Weights vs Base Weights",
                     fontweight="bold", fontsize=13)
    ax_vs1.grid(alpha=0.3)

    vol_recent = master["Market_Return"].rolling(VOL_SCALE_WINDOW).std() * np.sqrt(252) * 100
    ax_vs2.plot(master.index, vol_recent, color="#e74c3c",
                linewidth=1.2, label=f"Realized vol ({VOL_SCALE_WINDOW}d ann.)")
    ax_vs2.set_ylabel("Annualised Vol (%)"); ax_vs2.set_xlabel("Date")
    ax_vs2.legend(fontsize=9); ax_vs2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("volatility_scaled_weights.png", dpi=150)
    plt.show()
    print("  Saved: volatility_scaled_weights.png")
# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6b — TRANSITION SIGNAL SCAN
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 6b — Regime Transition Signal Scan")
print("=" * 60)

print("\nThreshold reference table:")
print(get_transition_summary().to_string(index=False))

transition_signals = scan_all_transitions(master, label_col="Resolved_Label")

if not transition_signals.empty:
    print(f"\nFirst 10 triggered dates:")
    print(transition_signals.head(10).to_string())
    transition_signals.to_csv("regime_transition_signals.csv")
    print("\nAll signals saved → regime_transition_signals.csv")

# Live check on the most recent day
latest_ret = master["Market_Return"].rolling(20).sum().iloc[-1] * 100
latest_regime = master["Resolved_Label"].iloc[-1]
live = check_transition_signal(latest_regime, latest_ret)
print(f"\nLive signal [{master.index[-1].date()}]:")
print(f"  Current regime : {latest_regime}")
print(f"  20d return     : {latest_ret:+.2f}%")
print(f"  Threshold      : {live['threshold_pct']:+.1f}%  →  {live['next_regime']}")
print(f"  Status         : {'⚠ TRANSITION SIGNAL ACTIVE' if live['triggered'] else '✓ No signal'}")
print("\nMaster data saved → regime_master_data.csv")

'''# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6c — MODEL ACCURACY VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 6c — Model Accuracy Validation")
print("=" * 60)

from sklearn.metrics import silhouette_score, calinski_harabasz_score

# 1. CLUSTER QUALITY METRICS
print("\n[1] Cluster Quality Metrics (HMM)")
hmm_clusters = master["HMM_Regime"]
features_for_eval = master[["Market_Return", "Volatility", "Momentum", "RSI", "Breadth"]]
X_eval = scaler.transform(features_for_eval)  # Use same scaler from training

silhouette = silhouette_score(X_eval, hmm_clusters)
calinski = calinski_harabasz_score(X_eval, hmm_clusters)

print(f"  Silhouette Score: {silhouette:.3f}")
print(f"    (Range: -1 to 1. >0.5 = strong, >0.25 = moderate)")
print(f"  Calinski-Harabasz Score: {calinski:.1f}")
print(f"    (Higher = better separated clusters)")

# 2. KNOWN EVENT DETECTION TEST
print("\n[2] Known Event Detection Test")
print("  Checking if model identifies major market events correctly...")

# Define known market events with date ranges and expected regimes
known_events = [
    ("COVID Crash", "2020-02-20", "2020-03-23", ["Despair", "Capitulation", "Pessimism"]),
    ("Vaccine Rally", "2020-11-09", "2021-02-15", ["Hope", "Optimism", "Enthusiasm"]),
    ("2022 Bear Market", "2022-01-01", "2022-06-30", ["Euphoria", "Unease", "Denial", "Pessimism"]),
    ("2023 Recovery", "2023-01-01", "2023-06-30", ["Hope", "Optimism"]),
    ("2024 Rally", "2024-01-01", "2024-12-31", ["Enthusiasm", "Exhilaration"]),
]

event_accuracy = []
for event_name, start_date, end_date, expected_regimes in known_events:
    try:
        mask = (master.index >= start_date) & (master.index <= end_date)
        if mask.sum() == 0:
            print(f"  ⚠ {event_name}: No data in range (your data starts {master.index.min().date()})")
            continue

        actual_regimes = master.loc[mask, "Resolved_Label"].unique()
        matches = [r for r in actual_regimes if r in expected_regimes]
        match_pct = len(matches) / len(expected_regimes) * 100

        event_accuracy.append(match_pct)
        status = "✓" if match_pct >= 50 else "✗"
        print(f"  {status} {event_name}: Found {matches}, Expected {expected_regimes} ({match_pct:.0f}% match)")
    except Exception as e:
        print(f"  ⚠ {event_name}: Error - {e}")

if event_accuracy:
    avg_event_accuracy = sum(event_accuracy) / len(event_accuracy)
    print(f"\n  Average Event Detection Accuracy: {avg_event_accuracy:.1f}%")
    print(f"    (>60% = good regime identification)")

# 3. REGIME CHARACTERISTICS VALIDATION
print("\n[3] Regime Characteristics Validation")
print("  Checking if each emotion has expected feature profile...")

regime_stats = master.groupby("Resolved_Label")[["Market_Return", "Volatility", "RSI", "Breadth"]].mean()

# Expected patterns: Bull states should have high return, low vol; Bear states opposite
validation_checks = []

# Check 1: Highest return regime should be Bull state (Enthusiasm/Exhilaration)
highest_return_regime = regime_stats["Market_Return"].idxmax()
if highest_return_regime in ["Enthusiasm", "Exhilaration", "Optimism"]:
    print(f"  ✓ Highest return regime: {highest_return_regime} (+{regime_stats.loc[highest_return_regime, 'Market_Return']*100:.2f}%/day)")
    validation_checks.append(True)
else:
    print(f"  ✗ Highest return regime: {highest_return_regime} (should be Bull state)")
    validation_checks.append(False)

# Check 2: Lowest return regime should be Bear state (Despair/Capitulation)
lowest_return_regime = regime_stats["Market_Return"].idxmin()
if lowest_return_regime in ["Despair", "Capitulation", "Pessimism"]:
    print(f"  ✓ Lowest return regime: {lowest_return_regime} ({regime_stats.loc[lowest_return_regime, 'Market_Return']*100:.2f}%/day)")
    validation_checks.append(True)
else:
    print(f"  ✗ Lowest return regime: {lowest_return_regime} (should be Bear state)")
    validation_checks.append(False)

# Check 3: Highest volatility should be in Bear/Turning states
highest_vol_regime = regime_stats["Volatility"].idxmax()
if highest_vol_regime in ["Despair", "Capitulation", "Pessimism", "Unease", "Denial"]:
    print(f"  ✓ Highest volatility regime: {highest_vol_regime} ({regime_stats.loc[highest_vol_regime, 'Volatility']*100:.2f}%/day)")
    validation_checks.append(True)
else:
    print(f"  ✗ Highest volatility regime: {highest_vol_regime} (should be Bear/Turning state)")
    validation_checks.append(False)

# Check 4: Lowest volatility should be in Bull states
lowest_vol_regime = regime_stats["Volatility"].idxmin()
if lowest_vol_regime in ["Optimism", "Enthusiasm", "Exhilaration"]:
    print(f"  ✓ Lowest volatility regime: {lowest_vol_regime} ({regime_stats.loc[lowest_vol_regime, 'Volatility']*100:.2f}%/day)")
    validation_checks.append(True)
else:
    print(f"  ✗ Lowest volatility regime: {lowest_vol_regime} (should be Bull state)")
    validation_checks.append(False)

print(f"\n  Validation Score: {sum(validation_checks)}/{len(validation_checks)} checks passed")

# 4. STABILITY METRIC
print("\n[4] Regime Stability Check")
regime_changes = (master["Resolved_Label"] != master["Resolved_Label"].shift()).sum()
total_days = len(master)
avg_regime_duration = total_days / regime_changes if regime_changes > 0 else total_days

print(f"  Total regime changes: {regime_changes} over {total_days} days")
print(f"  Average regime duration: {avg_regime_duration:.1f} days")
if 20 <= avg_regime_duration <= 200:
    print(f"  ✓ Regime duration is realistic")
    stability_score = "GOOD"
elif avg_regime_duration < 20:
    print(f"  ⚠ Regimes changing too frequently (noisy)")
    stability_score = "TOO NOISY"
else:
    print(f"  ⚠ Regimes lasting too long (sticky)")
    stability_score = "TOO STICKY"

# 5. BACKTEST QUALITY METRICS
print("\n[5] Backtest Quality Metrics")
strategy_ret = master["Strategy_Return"]
benchmark_ret = master["Benchmark_Return"]

# Sharpe ratios
sharpe_strategy = (strategy_ret.mean() / strategy_ret.std()) * np.sqrt(252) if strategy_ret.std() > 0 else 0
sharpe_benchmark = (benchmark_ret.mean() / benchmark_ret.std()) * np.sqrt(252) if benchmark_ret.std() > 0 else 0

print(f"  Strategy Sharpe: {sharpe_strategy:.2f}")
print(f"  Benchmark Sharpe: {sharpe_benchmark:.2f}")
print(f"  Alpha (Strategy - Benchmark): {(sharpe_strategy - sharpe_benchmark):.2f}")

# Information Ratio (active return / tracking error)
active_return = strategy_ret - benchmark_ret
tracking_error = active_return.std() * np.sqrt(252)
info_ratio = (active_return.mean() * 252) / tracking_error if tracking_error > 0 else 0
print(f"  Information Ratio: {info_ratio:.2f}")

# Win rate
win_rate = (strategy_ret > 0).sum() / len(strategy_ret) * 100
print(f"  Strategy Win Rate: {win_rate:.1f}%")

# Validate no suspiciously good results
if sharpe_strategy > 3.0:
    print(f"  ⚠ WARNING: Sharpe > 3.0 is suspicious — check for look-ahead bias")
elif sharpe_strategy < 0:
    print(f"  ⚠ WARNING: Negative Sharpe — strategy losing money")
else:
    print(f"  ✓ Sharpe ratio is realistic")

# 6. SUMMARY SCORE
print("\n" + "=" * 60)
print("OVERALL MODEL VALIDATION SUMMARY")
print("=" * 60)

scores = {
    "Cluster Quality (Silhouette)": min(silhouette, 0.5) / 0.5 * 100,  # Cap at 0.5
    "Event Detection": avg_event_accuracy if event_accuracy else 0,
    "Regime Characteristics": sum(validation_checks) / len(validation_checks) * 100,
    "Stability": min(100, max(0, 100 - abs(avg_regime_duration - 60) / 2)),  # Optimal ~60 days
    "Backtest Quality": min(100, max(0, 50 + sharpe_strategy * 20 + info_ratio * 10)),  # Composite
}

print("\n  Category Scores:")
for category, score in scores.items():
    bar = "█" * int(score / 10)
    print(f"    {category:.<30} {score:5.1f}% {bar}")

overall_score = sum(scores.values()) / len(scores)
print(f"\n  {'OVERALL SCORE':.<30} {overall_score:5.1f}%")

if overall_score >= 70:
    print("\n  ✓ MODEL QUALITY: GOOD — Ready for production")
elif overall_score >= 50:
    print("\n  ⚠ MODEL QUALITY: MODERATE — Needs refinement")
else:
    print("\n  ✗ MODEL QUALITY: POOR — Significant issues detected")

print("\n" + "=" * 60)
# ─────────────────────────────────────────────────────────────────────────────'''
DASHBOARD_CODE = '''
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(page_title="Nifty 50 — Emotion Cycle Regimes",
                   page_icon="📈", layout="wide")

st.markdown("""<style>
.main-header{font-size:2rem;font-weight:800;border-bottom:3px solid #e74c3c;padding-bottom:.3rem;margin-bottom:.5rem;}
</style>""", unsafe_allow_html=True)

REGIME_COLORS = {
    "Optimism":"#27ae60","Enthusiasm":"#2ecc71","Exhilaration":"#f1c40f",
    "Euphoria":"#e67e22","Unease":"#e74c3c","Denial":"#c0392b",
    "Pessimism":"#8e44ad","Despair":"#2c3e50","Capitulation":"#7f8c8d","Hope":"#3498db",
}

EMOTION_WEIGHTS = {
    "Optimism":     1.00,
    "Enthusiasm":   1.00,
    "Exhilaration": 1.00,
    "Euphoria":     0.80,   # reduce at peak (contrarian)
    "Unease":       0.60,
    "Denial":       0.50,
    "Pessimism":    0.30,
    "Despair":      0.60,
    "Capitulation": 0.80,
    "Hope":         1.00,
}

EMOTION_DESC = {
    "Optimism":     "Market rising steadily. Cautious positive sentiment.",
    "Enthusiasm":   "Strong uptrend. Broad participation across sectors.",
    "Exhilaration": "Rapid gains. High breadth, strong momentum.",
    "Euphoria":     "Market peak zone. Extreme bullishness — contrarian alert.",
    "Unease":       "First cracks appear. Volatility rising, breadth narrowing.",
    "Denial":       "Pullback seen as temporary. Investors holding positions.",
    "Pessimism":    "Sustained decline. Negative sentiment broadening.",
    "Despair":      "Market trough zone. Maximum pain and fear.",
    "Capitulation": "Panic selling climax. Smart money begins accumulating.",
    "Hope":         "Stabilisation. Early signs of recovery.",
}

@st.cache_data
def load():
    return pd.read_csv("regime_master_data.csv", index_col=0, parse_dates=True)

master = load()

st.sidebar.markdown("## Controls")
# ── Column metadata for every model ──────────────────────────────────────────
MODEL_META = {
    "Resolved (Best)": {"label":"Resolved_Label","regime":"HMM_Regime",
                        "prob_prefix":"HMM_P_R","prob_label":"HMM_Label"},
    "HMM":             {"label":"HMM_Label",     "regime":"HMM_Regime",
                        "prob_prefix":"HMM_P_R","prob_label":"HMM_Label"},
    "HSMM":            {"label":"HSMM_Label",    "regime":"HSMM_Regime",
                        "prob_prefix":"HSMM_P_R","prob_label":"HSMM_Label"},
    "GMM":             {"label":"GMM_Label",     "regime":"GMM_Regime",
                        "prob_prefix":"GMM_P_R","prob_label":"GMM_Label"},
    "KMeans":          {"label":"KMeans_Label",  "regime":"KMeans_Regime",
                        "prob_prefix":None,      "prob_label":"KMeans_Label"},
}

# Only show models whose label column exists in the loaded data
_order = ["Resolved (Best)", "HMM", "HSMM", "GMM", "KMeans"]
available_models = [m for m in _order if MODEL_META[m]["label"] in master.columns]

model_choice = st.sidebar.selectbox("Regime Model", available_models)
meta         = MODEL_META[model_choice]
label_col    = meta["label"]

if label_col not in master.columns:
    st.error(f"**{model_choice} not available** — `{label_col}` missing from data. "
             "Re-run tester.py with HSMM enabled.")
    st.stop()

date_range = st.sidebar.date_input(
    "Date Range",
    value=[master.index.min().date(), master.index.max().date()],
    min_value=master.index.min().date(), max_value=master.index.max().date())

df = master.loc[str(date_range[0]):str(date_range[1])] if len(date_range)==2 else master

st.markdown('<p class="main-header">📈 Nifty 50 — Wall Street Emotion Cycle Detector</p>',
            unsafe_allow_html=True)

cur_regime = df[label_col].iloc[-1] if not df.empty else "N/A"
cur_date   = df.index[-1].strftime("%d %b %Y") if not df.empty else ""
col_cur    = REGIME_COLORS.get(cur_regime, "#aaa")
weight_cur = EMOTION_WEIGHTS.get(cur_regime, 0.5)

c1,c2,c3,c4,c5 = st.columns(5)
with c1:
    st.markdown(f"""<div style="background:#f8f9fa;border-radius:12px;padding:1rem;border-left:4px solid {col_cur}">
    <b>Current State</b><br>
    <span style="background:{col_cur};color:white;padding:.2rem .8rem;border-radius:20px;font-weight:700;font-size:1.1rem">{cur_regime}</span>
    <br><small style="color:#888">{cur_date}</small></div>""", unsafe_allow_html=True)
with c2: st.metric("Equity Allocation", f"{weight_cur*100:.0f}%",
                    help="Strategy allocation in this regime")
with c3: st.metric("Trading Days", f"{len(df):,}")
with c4: st.metric("Active Regimes", df[label_col].nunique())
with c5:
    if "Strategy_Return" in df.columns:
        tot = np.exp(df["Strategy_Return"].sum()) - 1
        bh  = np.exp(df["Benchmark_Return"].sum()) - 1
        st.metric("Strategy vs B&H", f"{tot*100:.1f}%", delta=f"{(tot-bh)*100:.1f}% alpha")

if cur_regime in EMOTION_DESC:
    st.info(f"**{cur_regime}:** {EMOTION_DESC[cur_regime]}")
st.markdown("---")

if model_choice == "HSMM":
    st.caption(
        "ℹ️ **HSMM:** Duration-aware HMM. Uses per-state negative-binomial duration "
        "distributions to suppress abnormally short regime runs, producing more "
        "persistent and realistic labels than a standard HMM."
    )

tab1,tab2,tab3,tab4,tab5 = st.tabs(
    ["🗺️ Timeline","📊 Probabilities","🕐 Cycle Clock","🔬 Features","💼 Backtest"])

with tab1:
    st.subheader(f"Emotional Cycle Timeline — {model_choice}")
    cum_idx = np.exp(df["Market_Return"].cumsum()) * 100

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.78,0.22],
        vertical_spacing=0.03
    )

    # price line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=cum_idx,
            mode="lines",
            name="Nifty proxy",
            line=dict(color="#2c3e50", width=1.5),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Index: %{y:.2f}<extra></extra>"
        ),
        row=1, col=1
    )

    labels = df[label_col]

    # build contiguous regime runs
    spans = []
    prev = None
    sd = None

    for date, regime in labels.items():
        if regime != prev:
            if prev is not None:
                spans.append((sd, date, prev))
            prev, sd = regime, date

    if prev is not None:
        spans.append((sd, labels.index[-1], prev))

    y_top = float(cum_idx.max()) * 1.03

    # hoverable regime bands (one trace per span)
    for i, (x0, x1, regime) in enumerate(spans):
        # use inclusive end date so no 1-day gaps create white separators
        x0_ts = pd.Timestamp(x0)
        x1_ts = pd.Timestamp(x1)

        duration = max((x1_ts - x0_ts).days + 1, 1)
        mid = x0_ts + (x1_ts - x0_ts) / 2

        # restore state_id for hover tooltip — use the selected model's regime col
        _regime_col = meta["regime"]
        if _regime_col in df.columns:
            state_slice = df.loc[x0:x1, _regime_col].dropna()
            state_id    = int(state_slice.mode().iloc[0]) if not state_slice.empty else ""
        else:
            state_id = ""

        fig.add_trace(
            go.Bar(
                x=[mid],
                y=[y_top],
                width=[duration * 86400000],   # full span width in ms
                base=0,
                marker=dict(
                    color=REGIME_COLORS.get(regime, "grey"),
                    line=dict(width=0)         # REMOVE white borders
                    ),
                opacity=0.18,
                name=regime,
                legendgroup=regime,
                showlegend=(regime not in [t.name for t in fig.data]),
                customdata=[[regime, state_id, duration, pd.Timestamp(x0)]],
                hovertemplate=
                    "Date: %{customdata[3]|%Y-%m-%d}<br>"
                    "Regime: %{customdata[0]}<br>"
                    "State ID: %{customdata[1]}<br>"
                    "Duration: %{customdata[2]} days"
                    "<extra></extra>",
            ),
            row=1, col=1
        )

    # lower strip remains
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=[1]*len(df),
            marker=dict(
                color=[REGIME_COLORS.get(r,"grey") for r in labels],
                line=dict(width=0)     # remove white vertical separators
                ),
            showlegend=False,
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Regime: %{marker.color}<extra></extra>",
            name="Regime"
        ),
        row=2, col=1
    )
    fig.add_trace(go.Bar(x=df.index,y=[1]*len(df),
        marker_color=[REGIME_COLORS.get(r,"grey") for r in labels],
        showlegend=False,name="Regime"),row=2,col=1)
    fig.update_yaxes(title_text="Index Level",row=1)
    fig.update_yaxes(title_text="State",showticklabels=False,row=2)
    fig.update_layout(
        bargap=0,
        bargroupgap=0,
        height=560,
        template="plotly_white",
        title=f"Nifty 50 — {model_choice} Emotional Regimes",
        hovermode="x unified",
        barmode="overlay",
        legend=dict(
            orientation="h",
            y=1.05,
            itemclick="toggle",
            itemdoubleclick="toggleothers"
            )
        )

    st.plotly_chart(fig, use_container_width=True)

    runs,prev_r,cnt=[],None,0
    for r in labels:
        if r==prev_r: cnt+=1
        else:
            if prev_r: runs.append({"State":prev_r,"Days":cnt})
            prev_r,cnt=r,1
    if prev_r: runs.append({"State":prev_r,"Days":cnt})
    runs_df=(pd.DataFrame(runs).groupby("State")["Days"]
             .agg(["count","mean","median","max"])
             .rename(columns={"count":"Spells","mean":"Avg Days","median":"Median","max":"Max Days"})
             .round(1))
    st.markdown("**State Duration Statistics**")
    st.dataframe(runs_df,use_container_width=True)

with tab2:
    st.subheader("Regime Probabilities Over Time")
# Route caption + prefix to selected model
    _captions = {
        "Resolved (Best)": ("Probabilities are from the **HMM** model. "
                            "Resolved (Best) corrects labels via transition thresholds."),
        "HMM":   "Probabilities shown are from the **HMM** model.",
        "HSMM":  ("Probabilities shown are from the **HSMM** model "
                  "(forward-backward posteriors on the duration-aware HMM)."),
        "GMM":   "Probabilities shown are from the **GMM** model.",
        "KMeans":"KMeans does not produce probabilities.",
    }
    st.caption(_captions.get(model_choice, ""))

    prefix    = meta["prob_prefix"] or ""
    prob_cols = [c for c in df.columns if c.startswith(prefix)] if prefix else []

    _label_hint = meta["prob_label"] if meta["prob_label"] in df.columns else label_col

    if prob_cols:
        emotion_mapping = {}
        for i, prob_col in enumerate(prob_cols):
            dominant_mask = df[prob_cols].idxmax(axis=1) == prob_col
            if dominant_mask.any():
                emotion = df.loc[dominant_mask, _label_hint].mode()
                if len(emotion) > 0:
                    emotion_mapping[prob_col] = emotion.iloc[0]
                else:
                    emotion_mapping[prob_col] = f"Cluster {i}"
            else:
                emotion_mapping[prob_col] = f"Cluster {i}"
        fig2 = go.Figure()
        for col in prob_cols:
            emotion_name = emotion_mapping.get(col, col.replace(prefix, "State "))
            fig2.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    mode="lines",
                    name=emotion_name,
                    stackgroup="one",
                    line=dict(width=0.5),
                    marker_color=REGIME_COLORS.get(emotion_name, None)
                )
            )
        fig2.update_layout(
            height=380,
            template="plotly_white",
            title="Stacked Regime Probability",
            yaxis=dict(range=[0, 1], title="Probability"),
            legend=dict(orientation="h")
        )
        st.plotly_chart(fig2, use_container_width=True)
        latest = df[prob_cols].iloc[-1]
        labels_bar = [emotion_mapping.get(c, c.replace(prefix, "")) for c in latest.index]
        bar = go.Figure()
        bar.add_trace(go.Bar(
            x=labels_bar,
            y=latest.values * 100,
            marker_color=[REGIME_COLORS.get(l, "#888") for l in labels_bar]
        ))
        bar.update_layout(
            height=280,
            showlegend=False,
            title="Latest Day Regime Probabilities (%)",
            xaxis_title="Regime",
            yaxis_title="Probability (%)"
        )
        st.plotly_chart(bar, use_container_width=True)
    else:
        if model_choice == "KMeans":
            st.info("Probability columns not available for KMeans (non-probabilistic model).")
        else:
            st.warning(
                f"No probability columns found with prefix `{prefix}`. "
                "HSMM probability columns (`HSMM_P_R*`) are generated when the pipeline "
                "is run with HSMM enabled. Re-run `tester.py` to produce them."
            )

with tab3:
    st.subheader("Emotion Cycle Clock")
    counts=df[label_col].value_counts()
    pie_labels=list(counts.index)
    pie_vals=list(counts.values)
    pie_colors=[REGIME_COLORS.get(r,"#888") for r in pie_labels]

    cycle_order=["Optimism","Enthusiasm","Exhilaration","Euphoria","Unease",
                 "Denial","Pessimism","Despair","Capitulation","Hope"]
    ordered=[(r,counts.get(r,0)) for r in cycle_order if r in counts]

    fig3=go.Figure(go.Pie(
        labels=[r for r,v in ordered],
        values=[v for r,v in ordered],
        marker_colors=[REGIME_COLORS.get(r,"#888") for r,v in ordered],
        hole=0.45,
        textinfo="label+percent",
        sort=False,
    ))
    fig3.update_layout(height=500,template="plotly_white",
        title="Time spent in each emotional state",
        annotations=[dict(text="Cycle<br>Clock",x=0.5,y=0.5,font_size=14,showarrow=False)])
    st.plotly_chart(fig3,use_container_width=True)

    st.markdown("**Allocation weights by emotional state**")
    alloc_df=pd.DataFrame([
        {"State":r,"Equity Allocation":f"{w*100:.0f}%","Phase":
         "Bull" if w>=0.7 else "Recovery" if w>=0.3 else "Bear"}
        for r,w in EMOTION_WEIGHTS.items()
    ])
    st.dataframe(alloc_df,use_container_width=True)

with tab4:
    st.subheader("Feature Behaviour by Emotional State")
    feat_cols=["Market_Return","Volatility","Dispersion","Avg_Correlation",
               "Breadth","Momentum","RSI"]
    sel=st.selectbox("Feature",feat_cols)
    regimes_p=sorted(df[label_col].dropna().unique())
    fig4=go.Figure()
    for r in regimes_p:
        sub=df.loc[df[label_col]==r,sel].dropna()
        fig4.add_trace(go.Box(y=sub,name=r,
            marker_color=REGIME_COLORS.get(r,"grey"),boxmean="sd"))
    fig4.update_layout(height=440,template="plotly_white",
        title=f"{sel} across Emotional States",yaxis_title=sel)
    st.plotly_chart(fig4,use_container_width=True)

with tab5:
    st.subheader("Emotion Cycle Strategy vs Buy & Hold")
    if "Strategy_Return" not in df.columns:
        st.warning("Backtest data not found.")
    else:
        cum_s=np.exp(df["Strategy_Return"].cumsum())
        cum_b=np.exp(df["Benchmark_Return"].cumsum())
        fig5=go.Figure()
        fig5.add_trace(go.Scatter(x=df.index,y=cum_s,name="Emotion Strategy",
            line=dict(color="#27ae60",width=2)))
        fig5.add_trace(go.Scatter(x=df.index,y=cum_b,name="Buy & Hold",
            line=dict(color="#e74c3c",width=2,dash="dash")))
        fig5.update_layout(height=380,template="plotly_white",
            title="Cumulative Performance",
            yaxis_title="Portfolio Value (₹1→₹X)",legend=dict(orientation="h"))
        st.plotly_chart(fig5,use_container_width=True)

        def metrics(log_r):
            s=np.exp(log_r)-1; c=np.exp(log_r.sum())-1
            ny=len(s)/252; cagr=(1+c)**(1/max(ny,.01))-1
            sharpe=(s.mean()/s.std())*np.sqrt(252) if s.std()>0 else 0
            cw=(1+s).cumprod(); mdd=((cw-cw.cummax())/cw.cummax()).min()
            return {"Total Return":f"{c*100:.1f}%","CAGR":f"{cagr*100:.1f}%",
                    "Sharpe":f"{sharpe:.2f}","Max Drawdown":f"{mdd*100:.1f}%"}

        col1,col2=st.columns(2)
        with col1:
            st.markdown("**Emotion Cycle Strategy**")
            st.dataframe(pd.DataFrame(metrics(df["Strategy_Return"]),index=["Value"]).T,
                         use_container_width=True)
        with col2:
            st.markdown("**Buy & Hold**")
            st.dataframe(pd.DataFrame(metrics(df["Benchmark_Return"]),index=["Value"]).T,
                         use_container_width=True)

        monthly=(np.exp(df["Strategy_Return"].resample("ME").sum())-1)*100
        mon_df=pd.DataFrame({"Year":monthly.index.year,
                             "Month":monthly.index.strftime("%b"),
                             "Return":monthly.values}).pivot(index="Year",columns="Month",values="Return")
        month_order=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        mon_df=mon_df.reindex(columns=[m for m in month_order if m in mon_df.columns])
        heat=px.imshow(mon_df,text_auto=".1f",color_continuous_scale="RdYlGn",
                       zmin=-10,zmax=10,labels=dict(color="Return (%)"),
                       title="Monthly Strategy Returns (%)")
        heat.update_layout(height=max(200,len(mon_df)*35+100))
        st.plotly_chart(heat,use_container_width=True)

st.markdown("---")
st.markdown("<small>Built by Vidit Jain · Wall Street Emotion Cycle · 10 Regimes · "
            "HMM | HSMM | GMM | KMeans</small>",
            unsafe_allow_html=True)
'''

with open("regime_dashboard.py", "w", encoding="utf-8") as f:
    f.write(DASHBOARD_CODE)

print("\nDashboard written → regime_dashboard.py")
print("\n  Launch with:")
print("  streamlit run regime_dashboard.py")
print("\n" + "=" * 60)
print("ALL PHASES COMPLETE — 10 Emotional Cycle Regimes ✓")
print("=" * 60)



