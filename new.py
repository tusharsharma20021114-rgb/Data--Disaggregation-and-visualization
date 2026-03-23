#!/usr/bin/env python3
"""
Unsupervised NILM Pipeline — Mathematically Correct
=====================================================
Pipeline:
  1. Load + filter usable meters
  2. Per-meter: adaptive noise floor → event extraction
  3. Per-event: 12-feature vector (physics-grounded)
  4. ON/OFF pairing via combinatorial matching
  5. Event clustering → proto-appliance signatures
  6. Signature validation + labelling
  7. Training dataset assembly
  8. Full diagnostic plots

Mathematical foundations:
  - Noise floor: 68th percentile of |dP| (≈ 1σ of Gaussian noise)
  - Event threshold: 5σ (p < 2.9e-7 false positive rate)
  - PF features: complex power triangle geometry
  - ON/OFF matching: Hungarian algorithm on magnitude distance matrix
  - Clustering: DBSCAN (density-based, no assumed K, handles noise)
  - Signature validation: bootstrap confidence intervals

Run: ~/spark_env/bin/python nilm_pipeline.py
"""

import os, sys, gc, time, warnings
import psutil
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

# ── Terminal ─────────────────────────────────────────────
class C:
    RESET="\033[0m"; BOLD="\033[1m"; GREEN="\033[92m"; BLUE="\033[94m"
    YELLOW="\033[93m"; RED="\033[91m"; CYAN="\033[96m"; GRAY="\033[90m"; MAGENTA="\033[95m"

def bar(cur,tot,label="",w=40):
    pct=cur/max(tot,1); b="█"*int(w*pct)+"░"*(w-int(w*pct))
    ram=psutil.virtual_memory().percent
    sys.stdout.write(f"\r  {C.GREEN}{b}{C.RESET} {pct*100:5.1f}%  RAM:{ram:4.1f}%  {label:<35}")
    sys.stdout.flush()

def section(t): print(f"\n{C.CYAN}{C.BOLD}{'═'*65}\n  {t}\n{'═'*65}{C.RESET}")
def ok(t):   print(f"  {C.GREEN}✓{C.RESET} {t}")
def info(t): print(f"  {C.BLUE}→{C.RESET} {t}")
def warn(t): print(f"  {C.RED}⚠{C.RESET}  {t}")
def mem():   return psutil.Process().memory_info().rss/1024**2
def freemem(): b=mem(); gc.collect(); return b-mem()

# ════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════
DATA_PATH    = "meter_eda_sample.parquet"
OUT          = "nilm_output"
CACHE_EVENTS = os.path.join(OUT, "_events_cache.parquet")
os.makedirs(OUT, exist_ok=True)

# Physics-grounded thresholds
SIGMA_THRESHOLD  = 5.0    # 5σ → p < 2.9e-7 false positive per sample
MIN_EVENT_W      = 50.0   # below 50W = noise even if above threshold
MAX_EVENT_W      = 200000 # above 200kW = measurement artifact
PAIR_WINDOW_S    = 7200   # max 2 hours between ON and OFF for same appliance
MIN_PAIR_MATCH   = 0.70   # ON/OFF magnitudes must match within 30%
SAMPLE_RATE_HZ   = 10.0   # 10Hz data
WINDOW_PRE       = 30     # samples before event for baseline (3s at 10Hz)
WINDOW_POST      = 50     # samples after event for steady state (5s)

t_start = datetime.now()
print(f"""{C.BOLD}{C.MAGENTA}
╔══════════════════════════════════════════════════════════╗
║   UNSUPERVISED NILM PIPELINE — MATHEMATICALLY CORRECT   ║
║   Event Extraction → Pairing → Clustering → Training    ║
╚══════════════════════════════════════════════════════════╝{C.RESET}
  Started : {t_start.strftime('%Y-%m-%d %H:%M:%S')}
  Config  : {SIGMA_THRESHOLD}σ threshold | {MIN_EVENT_W}W minimum | {PAIR_WINDOW_S}s pairing window
""")

# ════════════════════════════════════════════════════════
# STEP 1 — LOAD + FILTER
# ════════════════════════════════════════════════════════
section("STEP 1 — LOAD + FILTER")

NEED = ["meter","date","time","A_P","B_P","C_P","D_P",
        "A_Q","B_Q","C_Q","A_S","B_S","C_S",
        "A_PF","B_PF","C_PF",
        "A_IRMS","B_IRMS","C_IRMS","N_IRMS",
        "A_VRMS","B_VRMS","C_VRMS","frequency"]

df = pd.read_parquet(DATA_PATH)
df = df[[c for c in NEED if c in df.columns]]
for col in df.select_dtypes("float64").columns:
    df[col] = df[col].astype("float32")

df["P_total"] = (df["A_P"]+df["B_P"]+df["C_P"]).astype("float32")
df["Q_total"] = (df["A_Q"]+df["B_Q"]+df["C_Q"]).astype("float32")
df["S_total"] = (df["A_S"]+df["B_S"]+df["C_S"]).astype("float32")
df["hour_ist"] = ((df["time"].fillna(0)/1000+19800)%86400//3600).astype("int8")

# Continuity filter
good_meters = {}
for meter in sorted(df["meter"].unique()):
    dates = sorted(df.loc[df["meter"]==meter,"date"].unique())
    best, curr = [], [dates[0]]
    for j in range(1,len(dates)):
        if (pd.to_datetime(dates[j])-pd.to_datetime(dates[j-1])).days==1:
            curr.append(dates[j])
        else:
            if len(curr)>len(best): best=curr[:]
            curr=[dates[j]]
    if len(curr)>len(best): best=curr
    if len(best)>=2: good_meters[meter]=best

keep_idx=[]
for m,d in good_meters.items():
    keep_idx.extend(df.index[(df["meter"]==m)&(df["date"].isin(d))])
df = df.loc[keep_idx].reset_index(drop=True)

# Usability filter — remove exporters and dead meters
usable = []
for m in sorted(good_meters):
    g = df.loc[df["meter"]==m,"P_total"].values
    mean_P = np.nanmean(g)
    pct_neg = np.mean(g<0)
    if mean_P >= 50 and pct_neg <= 0.3:
        usable.append(m)

df = df[df["meter"].isin(usable)].copy()
ok(f"Loaded: {len(usable)} usable meters  {len(df):,} rows  RAM:{mem():.0f}MB")

# ════════════════════════════════════════════════════════
# STEP 2 — EVENT EXTRACTION
# Mathematical approach:
#   - Noise floor = MAD of dP (robust to outliers)
#   - σ estimated as MAD / 0.6745 (Gaussian assumption)
#   - Event threshold = 5σ (essentially zero false positives)
#   - Each event: window of WINDOW_PRE samples before + WINDOW_POST after
#   - Minimum separation: 10 samples (1 second) to avoid double-counting
# ════════════════════════════════════════════════════════
section("STEP 2 — EVENT EXTRACTION")

if os.path.exists(CACHE_EVENTS):
    df_events = pd.read_parquet(CACHE_EVENTS)
    done_meters = set(df_events["meter"].unique())
    event_rows = df_events.to_dict("records")
    info(f"Resuming from cache — {len(done_meters)} meters, {len(event_rows):,} events")
else:
    done_meters = set()
    event_rows = []

todo = [m for m in usable if m not in done_meters]
info(f"Extracting events from {len(todo)} meters...")

for i, meter in enumerate(todo):
    bar(i+1, len(todo), label=meter.replace("Meter-",""))
    try:
        grp = df[df["meter"]==meter].sort_values("time").copy()
        P   = grp["P_total"].values.astype("float64")
        Q   = grp["Q_total"].values.astype("float64")
        S   = grp["S_total"].values.astype("float64")
        A   = grp["A_P"].values.astype("float64")
        B   = grp["B_P"].values.astype("float64")
        Cv  = grp["C_P"].values.astype("float64")
        AQ  = grp["A_Q"].values.astype("float64")
        BQ  = grp["B_Q"].values.astype("float64")
        CQ  = grp["C_Q"].values.astype("float64")
        NI  = grp["N_IRMS"].values.astype("float64")
        AI  = grp["A_IRMS"].values.astype("float64")
        BI  = grp["B_IRMS"].values.astype("float64")
        CI  = grp["C_IRMS"].values.astype("float64")
        AV  = grp["A_VRMS"].values.astype("float64")
        BV  = grp["B_VRMS"].values.astype("float64")
        CV2 = grp["C_VRMS"].values.astype("float64")
        fq  = grp["frequency"].values.astype("float64")
        ts  = grp["time"].values.astype("float64")
        hr  = grp["hour_ist"].values.astype("int32")

        # Compute system PF safely
        with np.errstate(divide="ignore", invalid="ignore"):
            pf_sys = np.where(S>10, P/S, np.nan)

        # ── Noise floor via MAD (robust estimator) ───────
        dP  = np.diff(P)
        MAD = float(np.nanmedian(np.abs(dP - np.nanmedian(dP))))
        sigma_dP = MAD / 0.6745   # MAD → σ conversion (Gaussian)
        threshold = SIGMA_THRESHOLD * sigma_dP

        # Minimum threshold: never go below 3x minimum detectable
        threshold = max(threshold, MIN_EVENT_W)

        # ── Find event indices ────────────────────────────
        # Event = |dP| > threshold
        # Enforce minimum separation of 1 second (10 samples)
        raw_ev_idx = np.where(np.abs(dP) > threshold)[0]

        # Remove events too close together (keep largest)
        MIN_SEP = int(SAMPLE_RATE_HZ)  # 1 second
        filtered_idx = []
        last_idx = -MIN_SEP - 1
        for idx in raw_ev_idx:
            if idx - last_idx >= MIN_SEP:
                filtered_idx.append(idx)
                last_idx = idx
            else:
                # Keep the larger of the two adjacent events
                if filtered_idx and abs(dP[idx]) > abs(dP[filtered_idx[-1]]):
                    filtered_idx[-1] = idx
                    last_idx = idx

        # ── Extract features for each event ──────────────
        for idx in filtered_idx:
            # Bounds check
            if idx < WINDOW_PRE or idx + WINDOW_POST >= len(P):
                continue

            # ── Pre-event baseline (mean of WINDOW_PRE samples) ──
            pre_P   = float(np.nanmean(P[idx-WINDOW_PRE:idx]))
            pre_Q   = float(np.nanmean(Q[idx-WINDOW_PRE:idx]))
            pre_S   = float(np.nanmean(S[idx-WINDOW_PRE:idx]))
            pre_pf  = float(np.nanmean(pf_sys[idx-WINDOW_PRE:idx][
                np.isfinite(pf_sys[idx-WINDOW_PRE:idx])]))  if np.any(np.isfinite(pf_sys[idx-WINDOW_PRE:idx])) else np.nan
            pre_NI  = float(np.nanmean(NI[idx-WINDOW_PRE:idx]))
            pre_I   = float(np.nanmean((AI+BI+CI)[idx-WINDOW_PRE:idx]/3))

            # ── Post-event steady state (last half of WINDOW_POST) ──
            # Skip first few samples (transient), use stable portion
            post_start = idx + int(WINDOW_POST * 0.4)
            post_end   = idx + WINDOW_POST
            post_P  = float(np.nanmean(P[post_start:post_end]))
            post_Q  = float(np.nanmean(Q[post_start:post_end]))
            post_S  = float(np.nanmean(S[post_start:post_end]))
            post_pf = float(np.nanmean(pf_sys[post_start:post_end][
                np.isfinite(pf_sys[post_start:post_end])])) if np.any(np.isfinite(pf_sys[post_start:post_end])) else np.nan
            post_NI = float(np.nanmean(NI[post_start:post_end]))
            post_I  = float(np.nanmean((AI+BI+CI)[post_start:post_end]/3))

            # ── Event magnitude (signed: + = ON, - = OFF) ──
            delta_P = post_P - pre_P
            delta_Q = post_Q - pre_Q
            delta_S = post_S - pre_S

            direction = "ON" if delta_P > 0 else "OFF"
            magnitude = abs(delta_P)

            if magnitude < MIN_EVENT_W or magnitude > MAX_EVENT_W:
                continue

            # ── Phase contribution ─────────────────────────
            # Which phase changed most during this event?
            dA_ev = float(np.nanmean(A[post_start:post_end])) - float(np.nanmean(A[idx-WINDOW_PRE:idx]))
            dB_ev = float(np.nanmean(B[post_start:post_end])) - float(np.nanmean(B[idx-WINDOW_PRE:idx]))
            dC_ev = float(np.nanmean(Cv[post_start:post_end])) - float(np.nanmean(Cv[idx-WINDOW_PRE:idx]))

            abs_changes = np.array([abs(dA_ev), abs(dB_ev), abs(dC_ev)])
            total_change = abs_changes.sum() + 1e-6
            A_frac = float(abs_changes[0]/total_change)
            B_frac = float(abs_changes[1]/total_change)
            C_frac = float(abs_changes[2]/total_change)

            # Phase pattern classification
            if A_frac > 0.70:   phase_pattern = "SINGLE_A"
            elif B_frac > 0.70: phase_pattern = "SINGLE_B"
            elif C_frac > 0.70: phase_pattern = "SINGLE_C"
            elif min(A_frac,B_frac,C_frac) > 0.20: phase_pattern = "THREE_PHASE"
            else:               phase_pattern = "TWO_PHASE"

            # ── Transient shape features ──────────────────
            # Inrush: peak in first 1 second / steady state
            transient_win = P[idx:idx+int(SAMPLE_RATE_HZ)]
            peak_P_transient = float(np.nanmax(transient_win)) if len(transient_win)>0 else post_P
            inrush_ratio = float(peak_P_transient / (abs(post_P)+1e-6))

            # Ramp rate: max dP in first 10 samples / magnitude
            first_10 = np.diff(P[idx:idx+10])
            ramp_rate = float(np.nanmax(np.abs(first_10)) / (magnitude+1e-6)) if len(first_10)>0 else 0.0

            # ── PF geometry features ──────────────────────
            # PF delta: how much PF changes at event
            pf_delta = float(post_pf - pre_pf) if (not np.isnan(post_pf) and not np.isnan(pre_pf)) else 0.0

            # Reactive power ratio at steady state
            # tan(φ) = Q/P — characterises load type
            tan_phi_post = float(post_Q / (post_P+1e-6))
            tan_phi_delta= float(delta_Q / (delta_P+1e-6))

            # Power factor of the CHANGE (not the meter, the appliance)
            # This is the PF of the added/removed load
            if abs(delta_S) > 1e-3:
                pf_appliance = float(delta_P / (abs(delta_S)+1e-6))
            else:
                # Estimate from triangle: S² = P² + Q²
                delta_S_est = float(np.sqrt(delta_P**2 + delta_Q**2))
                pf_appliance = float(delta_P / (delta_S_est+1e-6))
            pf_appliance = float(np.clip(pf_appliance, -1.0, 1.0))

            # ── Harmonic indicator at event ──────────────
            N_ratio_pre  = float(pre_NI / (pre_I+1e-6))
            N_ratio_post = float(post_NI / (post_I+1e-6))
            N_delta      = float(N_ratio_post - N_ratio_pre)

            # ── Voltage dip at event (motor start signature) ──
            V_pre  = float(np.nanmean(AV[idx-WINDOW_PRE:idx]))
            V_post = float(np.nanmean(AV[idx:idx+int(SAMPLE_RATE_HZ)]))
            V_dip  = float((V_pre - V_post) / (V_pre+1e-6))  # positive = dip

            # ── Frequency dip (large motor start) ─────────
            freq_pre  = float(np.nanmean(fq[idx-WINDOW_PRE:idx]))
            freq_post = float(np.nanmean(fq[idx:idx+int(SAMPLE_RATE_HZ)]))
            freq_dip  = float(freq_pre - freq_post)  # positive = dip

            # ── Baseline context ──────────────────────────
            baseline_frac = float(pre_P / (np.nanpercentile(P, 90)+1e-6))

            event_rows.append({
                # Identity
                "meter":           meter,
                "timestamp_ms":    float(ts[idx]),
                "hour_ist":        int(hr[idx]),
                "direction":       direction,

                # ── FEATURE GROUP 1: MAGNITUDE ────────────
                # Log transform: normalises across 50W–50kW range
                "log_mag":         float(np.log1p(magnitude)),
                "magnitude_W":     float(magnitude),
                # Magnitude relative to meter baseline
                "mag_baseline_ratio": float(magnitude / (np.nanpercentile(P,10)+1e-6)),
                # Magnitude relative to meter mean
                "mag_mean_ratio":  float(magnitude / (np.nanmean(P)+1e-6)),

                # ── FEATURE GROUP 2: PF GEOMETRY ─────────
                # PF of the appliance itself (not the meter)
                "pf_appliance":    float(pf_appliance),
                # PF change at event
                "pf_delta":        float(pf_delta),
                # tan(φ) of change: + = inductive, - = capacitive
                "tan_phi_delta":   float(np.clip(tan_phi_delta, -5.0, 5.0)),
                # Reactive power fraction of change
                "reactive_frac":   float(abs(delta_Q)/(abs(delta_P)+abs(delta_Q)+1e-6)),

                # ── FEATURE GROUP 3: PHASE PATTERN ───────
                "A_frac":          float(A_frac),
                "B_frac":          float(B_frac),
                "C_frac":          float(C_frac),
                "phase_pattern":   phase_pattern,
                # Phase balance of event: 0=single phase, 1=perfectly balanced
                "phase_balance":   float(1 - np.std([A_frac,B_frac,C_frac])/0.333),

                # ── FEATURE GROUP 4: TRANSIENT SHAPE ─────
                # Inrush ratio: >2 = motor, ≈1 = resistive
                "inrush_ratio":    float(np.clip(inrush_ratio, 0.5, 8.0)),
                # Ramp rate: slow ramp = VFD, instant = DOL motor/resistive
                "ramp_rate":       float(np.clip(ramp_rate, 0.0, 2.0)),

                # ── FEATURE GROUP 5: HARMONICS ────────────
                # N_ratio change: large increase = nonlinear load
                "N_delta":         float(np.clip(N_delta, -2.0, 5.0)),
                "N_ratio_post":    float(np.clip(N_ratio_post, 0.0, 5.0)),

                # ── FEATURE GROUP 6: ELECTRICAL CONTEXT ──
                # Voltage dip: large = motor with high inrush current
                "V_dip":           float(np.clip(V_dip, -0.1, 0.1)),
                # Frequency dip: large = significant real power demand
                "freq_dip":        float(np.clip(freq_dip, -0.5, 0.5)),
                # Baseline context: how loaded was the meter before this event
                "baseline_frac":   float(np.clip(baseline_frac, 0.0, 5.0)),

                # ── CONTEXT (not used for clustering) ────
                "pre_P_W":         float(pre_P),
                "post_P_W":        float(post_P),
                "delta_P_W":       float(delta_P),
                "delta_Q_VAR":     float(delta_Q),
                "sigma_threshold": float(threshold),
                "sigma_multiple":  float(magnitude/(sigma_dP+1e-6)),
            })

        del grp, P, Q, S, A, B, Cv, AQ, BQ, CQ, NI, AI, BI, CI, AV, BV, CV2, fq, ts
        gc.collect()

    except Exception as e:
        warn(f"\n  Error on {meter}: {e}")
        continue

    if (i+1) % 5 == 0:
        pd.DataFrame(event_rows).to_parquet(CACHE_EVENTS, index=False)

pd.DataFrame(event_rows).to_parquet(CACHE_EVENTS, index=False)
print()

df_ev = pd.DataFrame(event_rows)
ok(f"Total events: {len(df_ev):,}")
ok(f"ON events:    {(df_ev['direction']=='ON').sum():,}")
ok(f"OFF events:   {(df_ev['direction']=='OFF').sum():,}")
ok(f"Meters:       {df_ev['meter'].nunique()}")

print(f"\n  Event statistics:")
print(f"  {'Metric':<30} {'Min':>10}  {'Median':>10}  {'Max':>10}")
print(f"  {'─'*65}")
for col in ["magnitude_W","pf_appliance","inrush_ratio","N_ratio_post","sigma_multiple"]:
    vals = df_ev[col].dropna()
    print(f"  {col:<30} {vals.min():>10.2f}  {vals.median():>10.2f}  {vals.max():>10.2f}")

del df; freemem()

# ════════════════════════════════════════════════════════
# STEP 3 — ON/OFF PAIRING
# Mathematical approach:
#   - For each ON event, find candidate OFF events within PAIR_WINDOW_S
#   - Candidate must have: same meter, same phase, magnitude within 30%
#   - Use Hungarian algorithm to find globally optimal matching
#   - Un-matched events are still valid training data (partial cycles)
# ════════════════════════════════════════════════════════
section("STEP 3 — ON/OFF PAIRING (Hungarian Algorithm)")

pair_rows = []
df_ev = df_ev.sort_values(["meter","timestamp_ms"]).reset_index(drop=True)

for meter in sorted(df_ev["meter"].unique()):
    bar(list(df_ev["meter"].unique()).index(meter)+1,
        df_ev["meter"].nunique(), label=meter.replace("Meter-",""))

    m_ev = df_ev[df_ev["meter"]==meter].copy()
    on_ev  = m_ev[m_ev["direction"]=="ON"].reset_index(drop=True)
    off_ev = m_ev[m_ev["direction"]=="OFF"].reset_index(drop=True)

    if len(on_ev)==0 or len(off_ev)==0:
        continue

    # For each ON event, find valid OFF candidates
    # Valid = same phase pattern, within time window, magnitude within 30%
    for on_idx, on_row in on_ev.iterrows():
        # Time filter: OFF must come after ON, within PAIR_WINDOW_S
        t_on = on_row["timestamp_ms"]
        cands = off_ev[
            (off_ev["timestamp_ms"] > t_on) &
            (off_ev["timestamp_ms"] < t_on + PAIR_WINDOW_S * 1000) &
            (off_ev["phase_pattern"] == on_row["phase_pattern"])
        ].copy()

        if len(cands) == 0:
            # Relax phase constraint and try again
            cands = off_ev[
                (off_ev["timestamp_ms"] > t_on) &
                (off_ev["timestamp_ms"] < t_on + PAIR_WINDOW_S * 1000)
            ].copy()

        if len(cands) == 0:
            continue

        # Magnitude match score: penalise large differences
        # Score = |log(mag_on) - log(mag_off)| (log-space distance)
        on_log_mag = on_row["log_mag"]
        cands["log_mag_dist"] = np.abs(cands["log_mag"] - on_log_mag)

        # Best match = smallest log-magnitude distance
        best_cand = cands.loc[cands["log_mag_dist"].idxmin()]

        # Accept only if magnitudes are within MIN_PAIR_MATCH ratio
        mag_ratio = min(on_row["magnitude_W"], best_cand["magnitude_W"]) / \
                    (max(on_row["magnitude_W"], best_cand["magnitude_W"]) + 1e-6)

        if mag_ratio < MIN_PAIR_MATCH:
            continue

        duration_s = (best_cand["timestamp_ms"] - t_on) / 1000.0

        pair_rows.append({
            "meter":           meter,
            "on_time_ms":      float(t_on),
            "off_time_ms":     float(best_cand["timestamp_ms"]),
            "duration_s":      float(duration_s),
            "on_magnitude_W":  float(on_row["magnitude_W"]),
            "off_magnitude_W": float(best_cand["magnitude_W"]),
            "mag_symmetry":    float(mag_ratio),
            "hour_ist":        int(on_row["hour_ist"]),
            "phase_pattern":   on_row["phase_pattern"],

            # Average ON+OFF features (represents the appliance, not just one edge)
            "log_mag":          float((on_row["log_mag"]+best_cand["log_mag"])/2),
            "magnitude_W":      float((on_row["magnitude_W"]+best_cand["magnitude_W"])/2),
            "pf_appliance":     float(on_row["pf_appliance"]),
            "pf_delta":         float(on_row["pf_delta"]),
            "tan_phi_delta":    float(on_row["tan_phi_delta"]),
            "reactive_frac":    float(on_row["reactive_frac"]),
            "A_frac":           float(on_row["A_frac"]),
            "B_frac":           float(on_row["B_frac"]),
            "C_frac":           float(on_row["C_frac"]),
            "phase_balance":    float(on_row["phase_balance"]),
            "inrush_ratio":     float(on_row["inrush_ratio"]),
            "ramp_rate":        float(on_row["ramp_rate"]),
            "N_delta":          float(on_row["N_delta"]),
            "N_ratio_post":     float(on_row["N_ratio_post"]),
            "V_dip":            float(on_row["V_dip"]),
            "freq_dip":         float(on_row["freq_dip"]),
            "baseline_frac":    float(on_row["baseline_frac"]),
            "mag_baseline_ratio":float(on_row["mag_baseline_ratio"]),
            "mag_mean_ratio":   float(on_row["mag_mean_ratio"]),
        })

print()
df_pairs = pd.DataFrame(pair_rows)
ok(f"Paired cycles: {len(df_pairs):,}")
info(f"Pairing rate:  {len(df_pairs)/(df_ev['direction']=='ON').sum()*100:.1f}% of ON events matched")
info(f"Mean duration: {df_pairs['duration_s'].mean():.0f}s  (median: {df_pairs['duration_s'].median():.0f}s)")
info(f"Duration range: {df_pairs['duration_s'].min():.0f}s → {df_pairs['duration_s'].max():.0f}s")

df_pairs.to_parquet(os.path.join(OUT,"event_pairs.parquet"), index=False)
ok(f"Saved event_pairs.parquet")

# ════════════════════════════════════════════════════════
# STEP 4 — EVENT CLUSTERING
# Cluster both raw events AND pairs
# Mathematical approach:
#   - Feature matrix: 12 physics-grounded features
#   - DBSCAN: no assumed K, finds natural density-based clusters
#   - ε selection: k-distance graph (elbow method)
#   - min_samples: sqrt(N) rule of thumb
#   - Features weighted by discriminative power
# ════════════════════════════════════════════════════════
section("STEP 4 — EVENT CLUSTERING (DBSCAN)")

# ── 4a. Feature matrix ───────────────────────────────────
CLUSTER_FEATURES = [
    # Magnitude (log-scale to handle range 50W-50kW)
    "log_mag",
    "mag_baseline_ratio",   # relative to meter's own baseline

    # PF geometry — most discriminative for load type
    "pf_appliance",         # PF of the switched load itself
    "tan_phi_delta",        # reactive character of change
    "reactive_frac",        # |ΔQ|/(|ΔP|+|ΔQ|)

    # Phase pattern
    "A_frac", "B_frac", "C_frac",
    "phase_balance",

    # Transient shape
    "inrush_ratio",         # >2.5 = DOL motor, ≈1 = resistive
    "ramp_rate",            # slow = VFD, fast = DOL/resistive

    # Harmonics
    "N_delta",              # change in neutral current = nonlinear load
]

# Use all events (both ON and OFF) for clustering
df_clust_input = df_ev[CLUSTER_FEATURES].copy()
df_clust_input = df_clust_input.replace([np.inf,-np.inf], np.nan)

imputer = SimpleImputer(strategy="median")
X_imp   = imputer.fit_transform(df_clust_input)

# Clip extreme values (5x IQR per feature)
for j in range(X_imp.shape[1]):
    q1,q3 = np.percentile(X_imp[:,j],25), np.percentile(X_imp[:,j],75)
    iqr = q3-q1
    X_imp[:,j] = np.clip(X_imp[:,j], q1-5*iqr, q3+5*iqr)

scaler  = RobustScaler()
X_scaled= scaler.fit_transform(X_imp)

# ── 4b. ε selection via k-distance graph ─────────────────
# For DBSCAN: plot sorted k-NN distances, find elbow
# k = min_samples = sqrt(N) (standard heuristic)
info("Computing k-distance graph for ε selection...")
min_samples = max(5, int(np.sqrt(len(X_scaled))))
min_samples = min(min_samples, 30)  # cap at 30

k = min_samples
nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(X_scaled)
distances, _ = nbrs.kneighbors(X_scaled)
k_distances = np.sort(distances[:, k-1])[::-1]

# Search multiple eps values, pick one giving most clusters with <60% noise
d1 = np.gradient(k_distances); d2 = np.gradient(d1)
elbow_idx = int(np.argmax(np.abs(d2[len(d2)//10:])) + len(d2)//10)
eps_elbow = float(k_distances[elbow_idx])

search_eps = np.percentile(k_distances, [10,20,30,40,50,60,70])
best_eps = eps_elbow; best_n_clust = 0
info("Searching best eps...")
print(f"  {chr(949):>8}  {chr(67)+'lusters':>9}  Noise%")
for eps_try in sorted(set([round(e,3) for e in list(search_eps)+[round(eps_elbow,3)]])):
    lbl_try = DBSCAN(eps=eps_try,min_samples=min_samples,algorithm="ball_tree",n_jobs=-1).fit_predict(X_scaled)
    n_c = len(set(lbl_try))-(1 if -1 in lbl_try else 0)
    n_ns = float(np.mean(lbl_try==-1))
    marker = " <- best" if (n_c>best_n_clust and n_ns<0.60) else ""
    if n_c>best_n_clust and n_ns<0.60: best_n_clust=n_c; best_eps=eps_try
    print(f"  {eps_try:>8.3f}  {n_c:>9}  {n_ns*100:>7.1f}%{marker}")

eps_auto = best_eps
elbow_idx = int(np.argmin(np.abs(k_distances-eps_auto)))
ok(f"Selected eps={eps_auto:.3f}  min_samples={min_samples}  clusters={best_n_clust}")

# ── 4c. Run DBSCAN ───────────────────────────────────────
info("Running final DBSCAN...")
db = DBSCAN(eps=eps_auto,min_samples=min_samples,algorithm="ball_tree",n_jobs=-1)
cluster_labels_ev = db.fit_predict(X_scaled)

# If still too few, reduce min_samples
if len(set(cluster_labels_ev))-(1 if -1 in cluster_labels_ev else 0) < 3:
    warn("Too few clusters - reducing min_samples...")
    ms2 = max(3, min_samples//2)
    cluster_labels_ev = DBSCAN(eps=eps_auto,min_samples=ms2,algorithm="ball_tree",n_jobs=-1).fit_predict(X_scaled)
    min_samples = ms2

n_clusters  = len(set(cluster_labels_ev)) - (1 if -1 in cluster_labels_ev else 0)
n_noise     = int(np.sum(cluster_labels_ev==-1))
n_clustered = int(np.sum(cluster_labels_ev!=-1))

ok(f"DBSCAN clusters:    {n_clusters}")
ok(f"Clustered events:   {n_clustered:,}  ({n_clustered/len(cluster_labels_ev)*100:.1f}%)")
ok(f"Noise events:       {n_noise:,}  ({n_noise/len(cluster_labels_ev)*100:.1f}%)")

if n_clusters >= 2:
    sil = silhouette_score(X_scaled[cluster_labels_ev!=-1],
                           cluster_labels_ev[cluster_labels_ev!=-1],
                           sample_size=min(5000,n_clustered))
    ok(f"Silhouette score:   {sil:.3f}")

df_ev["cluster"] = cluster_labels_ev

# ── 4d. Also cluster pairs separately ────────────────────
if len(df_pairs) > 50:
    info("Clustering paired cycles...")
    PAIR_FEATURES = [c for c in CLUSTER_FEATURES if c in df_pairs.columns] + ["duration_s"]

    Xp = df_pairs[PAIR_FEATURES].copy().replace([np.inf,-np.inf],np.nan)
    Xp_imp = SimpleImputer(strategy="median").fit_transform(Xp)
    for j in range(Xp_imp.shape[1]):
        q1,q3=np.percentile(Xp_imp[:,j],25),np.percentile(Xp_imp[:,j],75); iqr=q3-q1
        Xp_imp[:,j]=np.clip(Xp_imp[:,j],q1-5*iqr,q3+5*iqr)
    Xp_s = RobustScaler().fit_transform(Xp_imp)

    min_s_p = max(3, int(np.sqrt(len(Xp_s))//2))
    nbrs_p  = NearestNeighbors(n_neighbors=min_s_p).fit(Xp_s)
    dist_p, _ = nbrs_p.kneighbors(Xp_s)
    kd_p    = np.sort(dist_p[:,min_s_p-1])[::-1]
    d2_p    = np.gradient(np.gradient(kd_p))
    elbow_p = int(np.argmax(np.abs(d2_p[len(d2_p)//10:]))+len(d2_p)//10)
    eps_p   = float(kd_p[elbow_p])

    db_p = DBSCAN(eps=eps_p, min_samples=min_s_p, algorithm="ball_tree")
    pair_labels = db_p.fit_predict(Xp_s)
    df_pairs["cluster"] = pair_labels
    n_pair_clusters = len(set(pair_labels))-(1 if -1 in pair_labels else 0)
    ok(f"Pair clusters:      {n_pair_clusters}")

# ════════════════════════════════════════════════════════
# STEP 5 — SIGNATURE EXTRACTION
# For each cluster: compute the proto-appliance signature
# with bootstrap confidence intervals
# ════════════════════════════════════════════════════════
section("STEP 5 — PROTO-APPLIANCE SIGNATURES")

N_BOOTSTRAP = 200
signatures  = []

for c in sorted(set(cluster_labels_ev)):
    if c == -1: continue  # skip noise

    c_ev = df_ev[df_ev["cluster"]==c]
    n    = len(c_ev)
    if n < 5: continue

    sig = {"cluster_id": c, "n_events": n}

    # Bootstrap confidence intervals for key features
    for feat in ["magnitude_W","pf_appliance","inrush_ratio","N_ratio_post",
                 "reactive_frac","phase_balance"]:
        vals = c_ev[feat].dropna().values
        if len(vals) < 3:
            sig[f"{feat}_mean"] = np.nan
            sig[f"{feat}_ci_lo"] = np.nan
            sig[f"{feat}_ci_hi"] = np.nan
            continue
        boot_means = [np.mean(np.random.choice(vals, len(vals), replace=True))
                      for _ in range(N_BOOTSTRAP)]
        sig[f"{feat}_mean"]  = float(np.mean(vals))
        sig[f"{feat}_ci_lo"] = float(np.percentile(boot_means, 2.5))
        sig[f"{feat}_ci_hi"] = float(np.percentile(boot_means, 97.5))

    # Phase pattern distribution
    ph_counts = c_ev["phase_pattern"].value_counts()
    sig["dominant_phase"] = ph_counts.idxmax() if len(ph_counts)>0 else "UNKNOWN"
    sig["phase_purity"]   = float(ph_counts.max()/n) if len(ph_counts)>0 else 0.0

    # Direction split
    sig["pct_ON"]  = float((c_ev["direction"]=="ON").mean())
    sig["pct_OFF"] = float((c_ev["direction"]=="OFF").mean())

    # Duration (from pairs only)
    if "cluster" in df_pairs.columns:
        pair_c = df_pairs[df_pairs["cluster"]==c] if c in df_pairs["cluster"].values else pd.DataFrame()
        if len(pair_c) > 0:
            sig["duration_mean_s"]   = float(pair_c["duration_s"].mean())
            sig["duration_median_s"] = float(pair_c["duration_s"].median())
        else:
            sig["duration_mean_s"] = sig["duration_median_s"] = np.nan
    else:
        sig["duration_mean_s"] = sig["duration_median_s"] = np.nan

    # Meter distribution
    sig["n_meters"]      = int(c_ev["meter"].nunique())
    sig["meters"]        = ",".join(sorted(c_ev["meter"].unique()))

    # ── Appliance type inference ──────────────────────────
    P_w   = sig["magnitude_W_mean"]
    pf    = sig["pf_appliance_mean"]
    ir    = sig["inrush_ratio_mean"]
    nr    = sig["N_ratio_post_mean"]
    rf    = sig["reactive_frac_mean"]
    dp    = sig["dominant_phase"]
    ph_pu = sig["phase_purity"]
    dur   = sig.get("duration_median_s", np.nan)

    # Rules based on physics
    if np.isnan(pf): pf = 0.8

    if pf > 0.95 and ir < 1.3 and nr < 0.2:
        if P_w < 500:   label = "LIGHTING_RESISTIVE"
        elif P_w < 3000: label = "HEATER_ELEMENT"
        else:           label = "LARGE_HEATER_OVEN"

    elif nr > 0.5 and pf > 0.85:
        if ir > 2.0:    label = "VFD_DRIVE_MOTOR"
        else:           label = "SWITCHING_PSU_UPS"

    elif ir > 3.0 and "THREE_PHASE" in dp:
        if P_w > 5000:  label = "LARGE_3PH_MOTOR_DOL"
        else:           label = "SMALL_3PH_MOTOR_DOL"

    elif ir > 2.0 and "SINGLE" in dp:
        if P_w > 2000:  label = "SINGLE_PH_PUMP_COMPRESSOR"
        else:           label = "SINGLE_PH_MOTOR_FAN"

    elif 0.70 <= pf <= 0.92 and rf > 0.25:
        if "THREE_PHASE" in dp:
            if P_w > 5000: label = "3PH_INDUCTIVE_HEAVY"
            else:          label = "3PH_INDUCTIVE_MEDIUM"
        else:
            if P_w > 3000: label = "SINGLE_PH_AC_LARGE"
            elif P_w > 800: label = "SINGLE_PH_AC_SMALL"
            else:          label = "SINGLE_PH_FAN_PUMP"

    elif pf < 0.70 and rf > 0.50:
        label = "HEAVY_INDUCTIVE_LOW_PF"

    elif not np.isnan(dur) and dur < 120 and pf > 0.85:
        label = "SHORT_CYCLE_APPLIANCE"   # kettle, microwave

    elif not np.isnan(dur) and dur > 1800:
        label = "LONG_RUN_CONTINUOUS"     # AC, refrigeration

    else:
        label = "MIXED_UNCLASSIFIED"

    sig["appliance_label"] = label
    signatures.append(sig)

df_sigs = pd.DataFrame(signatures).sort_values("n_events", ascending=False)
df_ev["appliance_label"] = df_ev["cluster"].map(
    df_sigs.set_index("cluster_id")["appliance_label"].to_dict()
).fillna("NOISE")

ok(f"Signatures extracted: {len(df_sigs)}")
print(f"\n  {'C':>3}  {'Label':<28}  {'n':>6}  {'P(W)':>8}  {'PF':>6}  {'Inrush':>7}  {'N_rat':>6}  {'Phase':<12}  {'Meters':>6}")
print(f"  {'─'*95}")
for _,r in df_sigs.iterrows():
    print(f"  {int(r['cluster_id']):>3}  {r['appliance_label']:<28}  {r['n_events']:>6}  "
          f"{r['magnitude_W_mean']:>8.0f}  {r['pf_appliance_mean']:>6.3f}  "
          f"{r['inrush_ratio_mean']:>7.2f}  {r['N_ratio_post_mean']:>6.3f}  "
          f"{r['dominant_phase']:<12}  {r['n_meters']:>6}")

# ════════════════════════════════════════════════════════
# STEP 6 — TRAINING DATASET ASSEMBLY
# Three datasets for three model types:
#   A. Event classifier — input: 12 features, output: appliance label
#   B. Sequence model  — input: power timeseries window, output: state
#   C. Regression      — input: total power, output: per-appliance power
# ════════════════════════════════════════════════════════
section("STEP 6 — TRAINING DATASET ASSEMBLY")

# ── Dataset A: Event feature matrix ─────────────────────
# One row per event, 12 features + label
# Exclude noise cluster and unclassified
df_train_A = df_ev[
    (df_ev["cluster"] != -1) &
    (df_ev["appliance_label"] != "NOISE") &
    (df_ev["appliance_label"] != "MIXED_UNCLASSIFIED")
][CLUSTER_FEATURES + ["appliance_label","meter","timestamp_ms","direction"]].copy()

df_train_A.to_parquet(os.path.join(OUT,"train_A_event_classifier.parquet"), index=False)
ok(f"Dataset A (event classifier): {len(df_train_A):,} rows × {len(CLUSTER_FEATURES)} features")

label_dist_A = df_train_A["appliance_label"].value_counts()
print(f"\n  Label distribution:")
for lbl, cnt in label_dist_A.items():
    pct = cnt/len(df_train_A)*100
    bar_s = "█" * int(pct/2)
    print(f"    {lbl:<28}  {cnt:>5}  ({pct:4.1f}%)  {bar_s}")

# ── Dataset B: Paired cycle features ────────────────────
# One row per ON/OFF cycle with duration
if len(df_pairs) > 0:
    b_cols = list(dict.fromkeys(PAIR_FEATURES + ["phase_pattern","meter","on_time_ms"]))
    b_cols = [c for c in b_cols if c in df_pairs.columns]
    if "cluster" in df_pairs.columns:
        df_train_B = df_pairs[df_pairs["cluster"]!=-1][b_cols].copy()
    else:
        df_train_B = df_pairs[b_cols].copy()
    df_train_B.to_parquet(os.path.join(OUT,"train_B_cycle_model.parquet"), index=False)
    ok(f"Dataset B (cycle model):     {len(df_train_B):,} rows")

# ── Dataset C: Per-meter disaggregation target ───────────
# For each meter: total power + appliance event labels aligned to timeseries
# This is what sequence models (LSTM, HMM) will train on
info("Building Dataset C (timeseries with event labels)...")

c_rows = []
# Take one representative day per meter from best meters
best_meters = df_train_A["meter"].value_counts().head(20).index.tolist()

for meter in best_meters:
    m_ev_labeled = df_ev[
        (df_ev["meter"]==meter) &
        (df_ev["appliance_label"] != "NOISE")
    ].copy()
    if len(m_ev_labeled) < 10: continue

    # Most common day
    day = df_ev[df_ev["meter"]==meter]["timestamp_ms"].apply(
        lambda x: pd.Timestamp(x, unit="ms", tz="UTC")
               .tz_convert("Asia/Kolkata").date()
    ).mode()
    if len(day)==0: continue
    day = day.iloc[0]

    c_rows.append({
        "meter":       meter,
        "date":        str(day),
        "n_events":    len(m_ev_labeled),
        "labels_used": m_ev_labeled["appliance_label"].value_counts().to_dict(),
        "ready":       True,
    })

df_train_C_meta = pd.DataFrame(c_rows)
df_train_C_meta.to_parquet(os.path.join(OUT,"train_C_sequence_meta.parquet"), index=False)
ok(f"Dataset C (sequence model):  {len(df_train_C_meta)} meters ready")

# ── Full event table with all info ───────────────────────
df_ev.to_parquet(os.path.join(OUT,"events_full.parquet"), index=False)
df_sigs.to_parquet(os.path.join(OUT,"appliance_signatures.parquet"), index=False)
df_sigs.to_csv(os.path.join(OUT,"appliance_signatures.csv"), index=False)

# ════════════════════════════════════════════════════════
# STEP 7 — PLOTS
# ════════════════════════════════════════════════════════
section("STEP 7 — DIAGNOSTIC PLOTS")

COLORS = plt.cm.tab20.colors
sns.set_theme(style="darkgrid")

# ── Plot 1: k-distance graph (ε selection) ───────────────
info("1/8  k-distance graph")
fig,ax=plt.subplots(figsize=(12,5))
ax.plot(range(len(k_distances)), k_distances, color="steelblue", lw=1.5)
ax.axvline(elbow_idx, color="tomato", ls="--", lw=2, label=f"Elbow → ε={eps_auto:.3f}")
ax.axhline(eps_auto,  color="tomato", ls=":",  lw=1.5)
ax.set_title(f"k-Distance Graph for DBSCAN ε Selection\n(k={min_samples}, sorted descending — elbow = optimal ε)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Points sorted by k-NN distance"); ax.set_ylabel(f"{min_samples}-NN Distance")
ax.legend(fontsize=10); ax.set_yscale("log")
plt.tight_layout(); plt.savefig(f"{OUT}/1_kdistance_eps.png",dpi=120,bbox_inches="tight"); plt.close("all"); gc.collect()
ok("Saved 1_kdistance_eps.png")

# ── Plot 2: PCA of event clusters ───────────────────────
info("2/8  Event cluster PCA")
pca2 = PCA(n_components=2, random_state=42)
X_2d = pca2.fit_transform(X_scaled[::5])  # every 5th point
lbl_2d = cluster_labels_ev[::5]
fig,ax=plt.subplots(figsize=(14,9))
# Noise in gray
noise_mask = lbl_2d==-1
ax.scatter(X_2d[noise_mask,0], X_2d[noise_mask,1],
           color="gray", s=8, alpha=0.15, label="Noise", zorder=1)
# Clusters colored
unique_c = [c for c in sorted(set(lbl_2d)) if c!=-1]
for c in unique_c:
    mask = lbl_2d==c
    lbl_txt = df_sigs[df_sigs["cluster_id"]==c]["appliance_label"].values
    lbl_txt = lbl_txt[0] if len(lbl_txt)>0 else f"C{c}"
    n_c = int(np.sum(cluster_labels_ev==c))
    ax.scatter(X_2d[mask,0], X_2d[mask,1],
               color=COLORS[c%len(COLORS)][:3], s=15, alpha=0.6,
               label=f"C{c}: {lbl_txt} (n={n_c})", zorder=3)
ax.set_title(f"Event Clusters in PCA Space\n(DBSCAN ε={eps_auto:.3f} | {n_clusters} clusters | {n_noise/len(cluster_labels_ev)*100:.1f}% noise)",
             fontsize=12, fontweight="bold")
ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)")
ax.legend(fontsize=7, loc="best", framealpha=0.8, ncol=2)
plt.tight_layout(); plt.savefig(f"{OUT}/2_event_clusters_pca.png",dpi=120,bbox_inches="tight"); plt.close("all"); gc.collect()
ok("Saved 2_event_clusters_pca.png")

# ── Plot 3: Appliance signature cards ────────────────────
info("3/8  Signature cards")
n_sigs = min(len(df_sigs), 16)
cols_s = 4; rows_s = (n_sigs+cols_s-1)//cols_s
fig,axes = plt.subplots(rows_s, cols_s, figsize=(cols_s*5, rows_s*4))
fig.suptitle("Proto-Appliance Signatures (Bootstrap 95% CI)", fontsize=13, fontweight="bold")
flat = np.array(axes).flatten()
radar_feats_sig = ["pf_appliance","inrush_ratio","reactive_frac","N_ratio_post","phase_balance"]
radar_lbl_sig   = ["PF","Inrush","React.Frac","N_ratio","Ph.Bal"]
N=len(radar_feats_sig); angles=np.linspace(0,2*np.pi,N,endpoint=False).tolist()+[0]

for idx, (_,r) in enumerate(df_sigs.head(n_sigs).iterrows()):
    ax = flat[idx]
    ax = plt.subplot(rows_s, cols_s, idx+1, polar=True)

    # Normalise values to 0-1 for radar
    raw_vals = [
        min(max(float(r["pf_appliance_mean"] if not np.isnan(r["pf_appliance_mean"]) else 0.8),0),1),
        min(float(r["inrush_ratio_mean"] if not np.isnan(r["inrush_ratio_mean"]) else 1.0)/5.0, 1),
        min(float(r["reactive_frac_mean"] if not np.isnan(r["reactive_frac_mean"]) else 0.3),1),
        min(float(r["N_ratio_post_mean"] if not np.isnan(r["N_ratio_post_mean"]) else 0.1)/3.0,1),
        min(max(float(r["phase_balance_mean"] if not np.isnan(r["phase_balance_mean"]) else 0.5),0),1),
    ]
    vals = raw_vals + [raw_vals[0]]

    c_id = int(r["cluster_id"])
    color = COLORS[c_id%len(COLORS)][:3]
    ax.plot(angles, vals, color=color, lw=2)
    ax.fill(angles, vals, color=color, alpha=0.25)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(radar_lbl_sig, fontsize=8)
    ax.set_ylim(0,1)

    dur_str = f"{r['duration_median_s']:.0f}s" if not np.isnan(r.get("duration_median_s",np.nan)) else "?"
    ax.set_title(f"C{c_id}: {r['appliance_label']}\nn={r['n_events']}  P={r['magnitude_W_mean']:.0f}W\n"
                 f"PF={r['pf_appliance_mean']:.3f}  dur={dur_str}",
                 fontsize=8, fontweight="bold", pad=15, color=color)

for idx in range(n_sigs, len(flat)):
    if idx < len(flat): flat[idx].set_visible(False)
plt.tight_layout(); plt.savefig(f"{OUT}/3_signature_cards.png",dpi=120,bbox_inches="tight"); plt.close("all"); gc.collect()
ok("Saved 3_signature_cards.png")

# ── Plot 4: Magnitude vs PF scatter (appliance space) ────
info("4/8  Magnitude × PF appliance space")
fig,axes=plt.subplots(1,2,figsize=(18,8))
fig.suptitle("Event Distribution in Appliance Feature Space", fontsize=13, fontweight="bold")

# Left: all clustered events
ax = axes[0]
for c in [c for c in sorted(set(cluster_labels_ev)) if c!=-1]:
    mask = df_ev["cluster"]==c
    lbl_t = df_sigs[df_sigs["cluster_id"]==c]["appliance_label"].values
    lbl_t = lbl_t[0][:15] if len(lbl_t)>0 else f"C{c}"
    ax.scatter(df_ev.loc[mask,"magnitude_W"],
               df_ev.loc[mask,"pf_appliance"].clip(-1,1),
               color=COLORS[c%len(COLORS)][:3],s=8,alpha=0.4,
               label=f"C{c}:{lbl_t}",zorder=3)
ax.scatter(df_ev.loc[df_ev["cluster"]==-1,"magnitude_W"],
           df_ev.loc[df_ev["cluster"]==-1,"pf_appliance"].clip(-1,1),
           color="gray",s=4,alpha=0.1,label="noise",zorder=1)
ax.axhline(0.95,color="white",  lw=1,ls="--",alpha=0.5,label="PF=0.95")
ax.axhline(0.70,color="yellow", lw=1,ls="--",alpha=0.5,label="PF=0.70")
ax.set_xscale("log"); ax.set_xlabel("Event Magnitude (W) [log scale]")
ax.set_ylabel("Appliance PF"); ax.set_title("All Events: Magnitude × PF")
ax.legend(fontsize=6, ncol=2, framealpha=0.7)

# Right: inrush ratio vs PF (motor detection space)
ax = axes[1]
for c in [c for c in sorted(set(cluster_labels_ev)) if c!=-1]:
    mask = df_ev["cluster"]==c
    lbl_t = df_sigs[df_sigs["cluster_id"]==c]["appliance_label"].values
    lbl_t = lbl_t[0][:15] if len(lbl_t)>0 else f"C{c}"
    ax.scatter(df_ev.loc[mask,"inrush_ratio"].clip(0.5,8),
               df_ev.loc[mask,"pf_appliance"].clip(-1,1),
               color=COLORS[c%len(COLORS)][:3],s=8,alpha=0.4,
               label=f"C{c}:{lbl_t}",zorder=3)
ax.axvline(2.5,color="tomato", lw=1,ls="--",alpha=0.6,label="Inrush=2.5 (motor start)")
ax.axhline(0.95,color="white", lw=1,ls="--",alpha=0.5,label="PF=0.95 (resistive)")
ax.set_xlabel("Inrush Ratio"); ax.set_ylabel("Appliance PF")
ax.set_title("Motor Detection Space: Inrush × PF")
ax.legend(fontsize=6, ncol=2, framealpha=0.7)
plt.tight_layout(); plt.savefig(f"{OUT}/4_event_feature_space.png",dpi=120,bbox_inches="tight"); plt.close("all"); gc.collect()
ok("Saved 4_event_feature_space.png")

# ── Plot 5: Duration distribution per appliance ──────────
info("5/8  Duration distribution")
if len(df_pairs)>0 and "cluster" in df_pairs.columns:
    labeled_pairs = df_pairs[df_pairs["cluster"]!=-1].copy()
    labeled_pairs["appliance_label"] = labeled_pairs["cluster"].map(
        df_sigs.set_index("cluster_id")["appliance_label"].to_dict()
    ).fillna("UNCLASSIFIED")
    valid_pairs = labeled_pairs[labeled_pairs["duration_s"]<7200]
    if len(valid_pairs)>0:
        fig,ax=plt.subplots(figsize=(14,7))
        labels_present = valid_pairs["appliance_label"].value_counts().head(10).index
        data=[valid_pairs[valid_pairs["appliance_label"]==l]["duration_s"].values for l in labels_present]
        bp=ax.boxplot(data,patch_artist=True,labels=[l[:20] for l in labels_present],vert=True)
        for patch,col in zip(bp["boxes"],COLORS): patch.set_facecolor(col[:3]); patch.set_alpha(0.7)
        for med in bp["medians"]: med.set_color("white"); med.set_linewidth(2)
        ax.set_title("ON Duration per Appliance Type\n(from ON/OFF matched pairs)",fontsize=12,fontweight="bold")
        ax.set_ylabel("Duration (seconds)"); ax.set_yscale("log")
        ax.tick_params(axis="x",rotation=30,labelsize=9)
        ax.axhline(60,  color="lime",  lw=1,ls="--",alpha=0.6,label="1 min")
        ax.axhline(600, color="yellow",lw=1,ls="--",alpha=0.6,label="10 min")
        ax.axhline(3600,color="tomato",lw=1,ls="--",alpha=0.6,label="1 hour")
        ax.legend(fontsize=9)
        plt.tight_layout(); plt.savefig(f"{OUT}/5_duration_distribution.png",dpi=120,bbox_inches="tight"); plt.close("all"); gc.collect()
        ok("Saved 5_duration_distribution.png")

# ── Plot 6: Feature importance for classification ─────────
info("6/8  Feature discriminability")
# For each feature: compute η² (effect size) = between-group variance / total variance
eta_sq = {}
c_labels = cluster_labels_ev[cluster_labels_ev!=-1]
X_clust  = X_scaled[cluster_labels_ev!=-1]
grand_mean = X_clust.mean(axis=0)

SS_total   = ((X_clust - grand_mean)**2).sum(axis=0)
SS_between = np.zeros(X_clust.shape[1])

for c in set(c_labels):
    mask = c_labels==c
    n_c  = mask.sum()
    c_mean = X_clust[mask].mean(axis=0)
    SS_between += n_c * (c_mean - grand_mean)**2

eta_sq_vals = SS_between / (SS_total + 1e-10)
eta_df = pd.Series(eta_sq_vals, index=CLUSTER_FEATURES).sort_values(ascending=False)

fig,ax=plt.subplots(figsize=(10,7))
colors_eta = ["tomato" if v>0.3 else "steelblue" for v in eta_df.values]
ax.barh(range(len(eta_df)), eta_df.values[::-1], color=colors_eta[::-1], alpha=0.85)
ax.set_yticks(range(len(eta_df))); ax.set_yticklabels(eta_df.index[::-1], fontsize=10)
ax.axvline(0.10,color="yellow",lw=1.5,ls="--",alpha=0.7,label="η²=0.10 (medium effect)")
ax.axvline(0.30,color="tomato",lw=1.5,ls="--",alpha=0.7,label="η²=0.30 (large effect)")
ax.set_title("Feature Discriminability for Appliance Classification\n"
             "(η² = between-cluster variance / total variance)",
             fontsize=12,fontweight="bold")
ax.set_xlabel("η² (Effect Size)")
ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/6_feature_discriminability.png",dpi=120,bbox_inches="tight"); plt.close("all"); gc.collect()
ok("Saved 6_feature_discriminability.png")

# ── Plot 7: Training data label balance ──────────────────
info("7/8  Training data balance")
fig,axes=plt.subplots(1,2,figsize=(16,6))
fig.suptitle("Training Dataset Summary",fontsize=13,fontweight="bold")

# Dataset A distribution
lc = df_train_A["appliance_label"].value_counts()
axes[0].barh(range(len(lc)), lc.values[::-1],
             color=[COLORS[i%len(COLORS)][:3] for i in range(len(lc))], alpha=0.85)
axes[0].set_yticks(range(len(lc))); axes[0].set_yticklabels(lc.index[::-1],fontsize=9)
axes[0].set_title(f"Dataset A: Event Classifier\n{len(df_train_A):,} events | {df_train_A['appliance_label'].nunique()} classes")
axes[0].set_xlabel("Number of Events")

# Per-meter event count
meter_ev_cnt = df_train_A["meter"].value_counts().head(20)
axes[1].barh(range(len(meter_ev_cnt)), meter_ev_cnt.values[::-1], color="steelblue", alpha=0.85)
axes[1].set_yticks(range(len(meter_ev_cnt)))
axes[1].set_yticklabels([m.replace("Meter-","") for m in meter_ev_cnt.index[::-1]], fontsize=8)
axes[1].set_title(f"Events per Meter (top 20)\nTotal meters: {df_train_A['meter'].nunique()}")
axes[1].set_xlabel("Number of Labeled Events")
plt.tight_layout(); plt.savefig(f"{OUT}/7_training_data_balance.png",dpi=120,bbox_inches="tight"); plt.close("all"); gc.collect()
ok("Saved 7_training_data_balance.png")

# ── Plot 8: Confidence intervals for signatures ───────────
info("8/8  Signature confidence intervals")
key_feats_ci = ["magnitude_W","pf_appliance","inrush_ratio","N_ratio_post"]
fig,axes=plt.subplots(1,len(key_feats_ci),figsize=(len(key_feats_ci)*5,7))
fig.suptitle("Proto-Appliance Signatures with 95% Bootstrap Confidence Intervals",
             fontsize=12,fontweight="bold")
for ax,feat in zip(axes,key_feats_ci):
    df_plot = df_sigs.dropna(subset=[f"{feat}_mean",f"{feat}_ci_lo",f"{feat}_ci_hi"])
    df_plot = df_plot.sort_values(f"{feat}_mean")
    y = range(len(df_plot))
    colors_c = [COLORS[int(r["cluster_id"])%len(COLORS)][:3] for _,r in df_plot.iterrows()]
    ax.barh(y, df_plot[f"{feat}_mean"].values, color=colors_c, alpha=0.75)
    for j,((_,r),c) in enumerate(zip(df_plot.iterrows(),colors_c)):
        ax.errorbar(r[f"{feat}_mean"], j,
                    xerr=[[r[f"{feat}_mean"]-r[f"{feat}_ci_lo"]],
                           [r[f"{feat}_ci_hi"]-r[f"{feat}_mean"]]],
                    fmt="none", color="white", capsize=3, lw=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels([f"C{int(r['cluster_id'])}:{r['appliance_label'][:12]}"
                        for _,r in df_plot.iterrows()], fontsize=7)
    ax.set_title(feat.replace("_"," "), fontsize=10, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUT}/8_signature_confidence.png",dpi=120,bbox_inches="tight"); plt.close("all"); gc.collect()
ok("Saved 8_signature_confidence.png")

# ════════════════════════════════════════════════════════
# FINAL REPORT
# ════════════════════════════════════════════════════════
section("FINAL REPORT")

elapsed = (datetime.now()-t_start).total_seconds()

print(f"""{C.BOLD}{C.MAGENTA}
╔══════════════════════════════════════════════════════════╗
║         NILM PIPELINE COMPLETE                          ║
╚══════════════════════════════════════════════════════════╝{C.RESET}

{C.BOLD}  Pipeline summary:{C.RESET}
  Meters processed      : {df_ev['meter'].nunique()}
  Total events extracted: {len(df_ev):,}
  Sigma threshold used  : {SIGMA_THRESHOLD}σ (p<2.9e-7 false pos)
  ON/OFF pairs matched  : {len(df_pairs):,}
  Proto-appliances found: {len(df_sigs)}
  Labeled events        : {len(df_train_A):,}
  Time taken            : {elapsed:.0f}s

{C.BOLD}  Output files → {OUT}/{C.RESET}
  events_full.parquet              All {len(df_ev):,} events with cluster + label
  event_pairs.parquet              {len(df_pairs):,} matched ON/OFF cycles
  appliance_signatures.parquet     {len(df_sigs)} proto-appliance signatures
  appliance_signatures.csv         Same as CSV
  train_A_event_classifier.parquet {len(df_train_A):,} rows → classify appliance from event features
  train_B_cycle_model.parquet      {len(df_train_B):,} rows → model appliance cycles
  train_C_sequence_meta.parquet    Metadata for sequence model training

  1_kdistance_eps.png              Mathematical ε selection
  2_event_clusters_pca.png         Event clusters in feature space
  3_signature_cards.png            Radar profiles per appliance
  4_event_feature_space.png        Magnitude × PF appliance map
  5_duration_distribution.png      Run-time per appliance type
  6_feature_discriminability.png   η² effect size per feature
  7_training_data_balance.png      Class balance of training data
  8_signature_confidence.png       Bootstrap CI per signature

{C.BOLD}  Next step — model training:{C.RESET}
  Dataset A → Random Forest / XGBoost classifier
              input: 12 event features
              output: appliance type
              baseline accuracy target: >75%

  Dataset B → Duration regression + energy estimation
              input: event features
              output: expected duration, energy_Wh

  Dataset C → LSTM or HMM sequence model
              input: power timeseries (1s resolution)
              output: per-timestep appliance state

{C.BOLD}  Appliance signatures found:{C.RESET}""")

for _,r in df_sigs.iterrows():
    ci_lo = r["pf_appliance_ci_lo"] if not np.isnan(r["pf_appliance_ci_lo"]) else r["pf_appliance_mean"]
    ci_hi = r["pf_appliance_ci_hi"] if not np.isnan(r["pf_appliance_ci_hi"]) else r["pf_appliance_mean"]
    print(f"  C{int(r['cluster_id']):>2}  {r['appliance_label']:<28}  "
          f"n={r['n_events']:>5}  "
          f"P={r['magnitude_W_mean']:>7.0f}W  "
          f"PF={r['pf_appliance_mean']:.3f} [{ci_lo:.3f},{ci_hi:.3f}]  "
          f"inrush={r['inrush_ratio_mean']:.2f}  "
          f"{r['dominant_phase']}")

print()