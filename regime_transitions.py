# -*- coding: utf-8 -*-
"""
regime_transitions.py — BIDIRECTIONAL version
"""

import pandas as pd
import numpy as np


REGIME_SEQUENCE = [
    "Optimism", "Enthusiasm", "Exhilaration", "Euphoria",
    "Unease", "Denial",
    "Pessimism", "Despair", "Capitulation",
    "Hope",
]

REGIME_COLORS = {
    "Optimism":     "#27ae60",
    "Enthusiasm":   "#2ecc71",
    "Exhilaration": "#f1c40f",
    "Euphoria":     "#e67e22",
    "Unease":       "#e74c3c",
    "Denial":       "#c0392b",
    "Pessimism":    "#8e44ad",
    "Despair":      "#2c3e50",
    "Capitulation": "#7f8c8d",
    "Hope":         "#3498db",
}

REGIME_PHASE = {
    "Optimism"     : "Bull",
    "Enthusiasm"   : "Bull",
    "Exhilaration" : "Bull",
    "Euphoria"     : "Turning",
    "Unease"       : "Turning",
    "Denial"       : "Bear",
    "Pessimism"    : "Bear",
    "Despair"      : "Bear",
    "Capitulation" : "Recovery",
    "Hope"         : "Recovery",
}

REGIME_TRANSITION_THRESHOLDS = {
    "Optimism"     : ("Enthusiasm",   +4.0),
    "Enthusiasm"   : ("Exhilaration", +6.0),
    "Exhilaration" : ("Euphoria",     +8.0),
    "Euphoria"     : ("Unease",       -4.0),
    "Unease"       : ("Denial",       -6.0),
    "Denial"       : ("Pessimism",    -9.0),
    "Pessimism"    : ("Despair",      -12.0),
    "Despair"      : ("Capitulation", -5.0),
    "Capitulation" : ("Hope",         +3.0),
    "Hope"         : ("Optimism",     +5.0),
}


def _result(triggered, next_regime, threshold_pct, r, direction, phase):
    return {
        "triggered"    : triggered,
        "next_regime"  : next_regime,
        "threshold_pct": threshold_pct,
        "excess_pct"   : round(float(r - threshold_pct), 2),
        "direction"    : direction,
        "phase"        : phase,
    }


def check_transition_signal(current_regime: str,
                             rolling_return_pct: float) -> dict:
    r = rolling_return_pct

    # ── BULL PHASE ──
    if current_regime == "Optimism":
        if r >= +4.0:
            return _result(True,  "Enthusiasm",  +4.0, r, "forward",  "Bull")
        elif r <= -3.0:
            return _result(True,  "Hope",        -3.0, r, "backward", "Bull")
        else:
            return _result(False, "Enthusiasm",  +4.0, r, "forward",  "Bull")

    elif current_regime == "Enthusiasm":
        if r >= +6.0:
            return _result(True,  "Exhilaration", +6.0, r, "forward",  "Bull")
        elif r <= -4.0:
            return _result(True,  "Optimism",     -4.0, r, "backward", "Bull")
        else:
            return _result(False, "Exhilaration", +6.0, r, "forward",  "Bull")

    elif current_regime == "Exhilaration":
        if r >= +8.0:
            return _result(True,  "Euphoria",   +8.0, r, "forward",  "Bull")
        elif r <= -5.0:
            return _result(True,  "Enthusiasm", -5.0, r, "backward", "Bull")
        else:
            return _result(False, "Euphoria",   +8.0, r, "forward",  "Bull")

    # ── TURNING POINT ──
    elif current_regime == "Euphoria":
        if r <= -4.0:
            return _result(True,  "Unease",       -4.0, r, "forward",  "Turning")
        elif r >= +5.0:
            return _result(True,  "Exhilaration", +5.0, r, "backward", "Turning")
        else:
            return _result(False, "Unease",       -4.0, r, "forward",  "Turning")

    elif current_regime == "Unease":
        if r <= -6.0:
            return _result(True,  "Denial",   -6.0, r, "forward",  "Turning")
        elif r >= +4.0:
            return _result(True,  "Euphoria", +4.0, r, "backward", "Turning")
        else:
            return _result(False, "Denial",   -6.0, r, "forward",  "Turning")

    # ── BEAR PHASE ──
    elif current_regime == "Denial":
        if r <= -9.0:
            return _result(True,  "Pessimism", -9.0, r, "forward",  "Bear")
        elif r >= +5.0:
            return _result(True,  "Unease",    +5.0, r, "backward", "Bear")
        else:
            return _result(False, "Pessimism", -9.0, r, "forward",  "Bear")

    elif current_regime == "Pessimism":
        if r <= -12.0:
            return _result(True,  "Despair", -12.0, r, "forward",  "Bear")
        elif r >= +6.0:
            return _result(True,  "Denial",   +6.0, r, "backward", "Bear")
        else:
            return _result(False, "Despair", -12.0, r, "forward",  "Bear")

    elif current_regime == "Despair":
        if r <= -5.0:
            return _result(True,  "Capitulation", -5.0, r, "forward",  "Bear")
        elif r >= +4.0:
            return _result(True,  "Pessimism",    +4.0, r, "backward", "Bear")
        else:
            return _result(False, "Capitulation", -5.0, r, "forward",  "Bear")

    # ── RECOVERY PHASE ──
    elif current_regime == "Capitulation":
        if r >= +3.0:
            return _result(True,  "Hope",    +3.0, r, "forward",  "Recovery")
        elif r <= -3.0:
            return _result(True,  "Despair", -3.0, r, "backward", "Recovery")
        else:
            return _result(False, "Hope",    +3.0, r, "forward",  "Recovery")

    elif current_regime == "Hope":
        if r >= +5.0:
            return _result(True,  "Optimism",     +5.0, r, "forward",  "Recovery")
        elif r <= -4.0:
            return _result(True,  "Capitulation", -4.0, r, "backward", "Recovery")
        else:
            return _result(False, "Optimism",     +5.0, r, "forward",  "Recovery")

    else:
        return {"triggered": False, "next_regime": None, "threshold_pct": None,
                "excess_pct": 0.0, "direction": None, "phase": "Unknown"}


def get_transition_summary() -> pd.DataFrame:
    both = {
        "Optimism"     : ("Enthusiasm",   +4.0, "Hope",         -3.0),
        "Enthusiasm"   : ("Exhilaration", +6.0, "Optimism",     -4.0),
        "Exhilaration" : ("Euphoria",     +8.0, "Enthusiasm",   -5.0),
        "Euphoria"     : ("Unease",       -4.0, "Exhilaration", +5.0),
        "Unease"       : ("Denial",       -6.0, "Euphoria",     +4.0),
        "Denial"       : ("Pessimism",    -9.0, "Unease",       +5.0),
        "Pessimism"    : ("Despair",     -12.0, "Denial",       +6.0),
        "Despair"      : ("Capitulation", -5.0, "Pessimism",    +4.0),
        "Capitulation" : ("Hope",         +3.0, "Despair",      -3.0),
        "Hope"         : ("Optimism",     +5.0, "Capitulation", -4.0),
    }
    records = []
    for regime in REGIME_SEQUENCE:
        fwd_to, fwd_thr, bwd_to, bwd_thr = both[regime]
        records.append({
            "Regime": regime, "Phase": REGIME_PHASE[regime],
            "Forward_To": fwd_to, "Forward_Thr (%)": fwd_thr,
            "Backward_To": bwd_to, "Backward_Thr (%)": bwd_thr,
        })
    return pd.DataFrame(records)


def scan_all_transitions(master_df: pd.DataFrame,
                          label_col: str = "HMM_Label",
                          return_col: str = "Market_Return",
                          window: int = 5) -> pd.DataFrame:
    for col in [label_col, return_col]:
        if col not in master_df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(master_df.columns)}")

    rolling_ret_pct = master_df[return_col].rolling(window).sum() * 100
    results = []

    for date, row in master_df.iterrows():
        regime = row[label_col]
        ret    = rolling_ret_pct.loc[date]
        if pd.isna(regime) or pd.isna(ret):
            continue
        signal = check_transition_signal(str(regime), float(ret))
        if signal["triggered"]:
            results.append({
                "Date": date, "Current_Regime": regime,
                "Next_Regime": signal["next_regime"],
                "Rolling_Return_Pct": round(ret, 2),
                "Threshold_Pct": signal["threshold_pct"],
                "Excess_Pct": signal["excess_pct"],
                "Direction": signal["direction"],
                "Phase": signal["phase"],
            })

    if not results:
        print("  [regime_transitions] No transition signals found.")
        return pd.DataFrame(columns=["Date","Current_Regime","Next_Regime",
            "Rolling_Return_Pct","Threshold_Pct","Excess_Pct","Direction","Phase"])

    out = pd.DataFrame(results).set_index("Date")
    fwd = (out["Direction"] == "forward").sum()
    bwd = (out["Direction"] == "backward").sum()
    print(f"  [regime_transitions] {len(out)} signal(s): {fwd} forward, {bwd} backward.")
    return out


def resolve_regime(master_df: pd.DataFrame,
                   label_col: str = "HMM_Label",
                   return_col: str = "Market_Return",
                   window: int = 5) -> pd.Series:
    """
    Hybrid approach: Uses HMM_Label as primary signal, with transition thresholds
    as confirmation/override. This prevents getting stuck in one regime while
    still benefiting from threshold-based corrections.
    """
    for col in [label_col, return_col]:
        if col not in master_df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(master_df.columns)}")

    rolling_ret = master_df[return_col].rolling(window).sum() * 100
    resolved    = []
    current     = master_df[label_col].iloc[0]   # seed from HMM label on Day 1

    for date, row in master_df.iterrows():
        ret = rolling_ret.loc[date]
        hmm_label = row[label_col]  # Today's HMM prediction

        if not pd.isna(ret):
            # Check if HMM suggests a regime change
            if hmm_label != current:
                # HMM says regime changed — verify with transition threshold
                signal = check_transition_signal(current, float(ret))

                if signal["triggered"]:
                    # Threshold triggered — check if it agrees with HMM
                    if signal["next_regime"] == hmm_label:
                        # Both HMM and threshold agree — confirm transition
                        current = hmm_label
                    else:
                        # Threshold says different regime than HMM — trust threshold
                        current = signal["next_regime"]
                else:
                    # No threshold trigger, but HMM changed — trust HMM
                    current = hmm_label
            # If hmm_label == current, stay in current regime (no change)

        resolved.append(current)

    series = pd.Series(resolved, index=master_df.index, name="Resolved_Label")
    print(f"  [resolve_regime] Done. {series.nunique()} unique regimes detected.")

    # Print distribution comparison
    print(f"\n  Label distribution comparison:")
    print(f"    HMM_Label unique: {master_df[label_col].nunique()}")
    print(f"    Resolved_Label unique: {series.nunique()}")

    return series


if __name__ == "__main__":
    print("=" * 65)
    print("regime_transitions.py — bidirectional self-test")
    print("=" * 65)

    print("\n── Full Threshold Table ──")
    print(get_transition_summary().to_string(index=False))

    tests = [
        # (regime, ret, expect_triggered, expect_next, label)
        ("Optimism",     +5.0,  True,  "Enthusiasm",    "forward  bull"),
        ("Optimism",     +3.5,  False, "Enthusiasm",    "no trigger"),
        ("Optimism",     -3.5,  True,  "Hope",          "backward bull"),
        ("Optimism",     -2.0,  False, "Enthusiasm",    "no trigger"),
        ("Enthusiasm",   -4.5,  True,  "Optimism",      "backward bull"),
        ("Euphoria",     -4.2,  True,  "Unease",        "forward  turning"),
        ("Euphoria",     +5.5,  True,  "Exhilaration",  "backward turning"),
        ("Denial",       +5.5,  True,  "Unease",        "backward bear"),
        ("Pessimism",   -13.5,  True,  "Despair",       "forward  bear"),
        ("Capitulation", +3.1,  True,  "Hope",          "forward  recovery"),
        ("Hope",         -4.5,  True,  "Capitulation",  "backward recovery"),
        ("Hope",         -2.0,  False, "Optimism",      "no trigger"),
    ]

    print("\n── Signal Tests ──")
    all_pass = True
    for regime, ret, exp_trig, exp_next, label in tests:
        s = check_transition_signal(regime, ret)
        ok = (s["triggered"] == exp_trig and
              (not exp_trig or s["next_regime"] == exp_next))
        if not ok:
            all_pass = False
        icon = "✓" if ok else "✗ FAIL"
        print(f"  {icon}  [{label:22s}]  {regime:15s}  ret={ret:+5.1f}%  "
              f"→ triggered={str(s['triggered']):5s}  next={s['next_regime'] or 'None'}")

    print(f"\n{'All {len(tests)} tests passed.' if all_pass else 'SOME TESTS FAILED.'}")
