"""Meesho COD RTO Decision Workspace — Risk Diagnosis, Checkout Simulator, Unit Economics & RICE Roadmap.

Independent case study using deterministic synthetic order data (n = 25,000 orders scaled 3,200x
to an 80M monthly order book). Preserves risk_engine.py, queries.sql, generate_data.py, and all
underlying SQL/scorecard calculations.
"""
from __future__ import annotations

import html
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
os.chdir(BASE)

from risk_engine import (  # noqa: E402
    COD_SURCHARGE,
    CONFIRM_RTO_CUT,
    DROP,
    INTERVENTION,
    RTO_COST,
    SURCHARGE_RTO_CUT,
    TAKE_RATE,
    TAKEUP,
    TIER_CUTS,
    UPI_DISCOUNT,
    W,
    logit,
    score_profile,
    sigmoid,
)

# ---------------------------------------------------------------------------
# Constants & Preset Configurations
# ---------------------------------------------------------------------------
TIER_ORDER = ["Low", "Medium", "High", "Very High"]

TIER_META = {
    "Low": {
        "badge_class": "rto-badge--emerald",
        "status_text": "LOW RISK (<15% RTO)",
        "color": "#10B981",
        "Gate": "Frictionless COD",
    },
    "Medium": {
        "badge_class": "rto-badge--amber",
        "status_text": "MEDIUM RISK (15–35% RTO)",
        "color": "#F59E0B",
        "Gate": "WhatsApp Confirmation",
    },
    "High": {
        "badge_class": "rto-badge--amber",
        "status_text": "HIGH RISK (35–60% RTO)",
        "color": "#F97316",
        "Gate": "Rs.40 UPI Incentive / Rs.30 COD Fee",
    },
    "Very High": {
        "badge_class": "rto-badge--red",
        "status_text": "VERY HIGH RISK (>=60% RTO)",
        "color": "#EF4444",
        "Gate": "Prepaid-Only Gate (Rs.40 UPI Discount)",
    },
}

SIMULATOR_DEFAULTS = {
    "sim_city_tier": 3,
    "sim_category": "Fashion",
    "sim_aov": 550,
    "sim_buyer_vintage": "First-time",
    "sim_payment_intent": "COD",
    "sim_seller_new": True,
    "sim_pin": 0.35,
}

SIMULATOR_PRESETS = {
    "Focus Segment (High Risk Default)": {
        "sim_city_tier": 3,
        "sim_category": "Fashion",
        "sim_aov": 550,
        "sim_buyer_vintage": "First-time",
        "sim_payment_intent": "COD",
        "sim_seller_new": True,
        "sim_pin": 0.35,
        "desc": "First-time Tier-3 Fashion COD buyer in mid-AOV band (Rs.550) with low PIN deliverability (0.35) -> High Risk (47/100).",
    },
    "Extreme Tail Risk (Very High Gate)": {
        "sim_city_tier": 4,
        "sim_category": "Fashion",
        "sim_aov": 600,
        "sim_buyer_vintage": "First-time",
        "sim_payment_intent": "COD",
        "sim_seller_new": True,
        "sim_pin": 0.10,
        "desc": "First-time Tier-4 Fashion COD buyer from an unverified seller in a weak delivery PIN (0.10) -> Very High Risk (64/100).",
    },
    "Repeat Regional COD (Medium Nudge)": {
        "sim_city_tier": 3,
        "sim_category": "Fashion",
        "sim_aov": 550,
        "sim_buyer_vintage": "Repeat",
        "sim_payment_intent": "COD",
        "sim_seller_new": True,
        "sim_pin": 0.15,
        "desc": "Repeat Tier-3 Fashion COD buyer ordering from a new seller in a low-deliverability PIN (0.15) -> Medium Risk (19/100).",
    },
    "Metro Prepaid Baseline (Low Frictionless)": {
        "sim_city_tier": 1,
        "sim_category": "Home",
        "sim_aov": 1500,
        "sim_buyer_vintage": "Repeat",
        "sim_payment_intent": "Prepaid (UPI)",
        "sim_seller_new": False,
        "sim_pin": 0.90,
        "desc": "Repeat Tier-1 Home buyer paying upfront via UPI in a high-deliverability PIN (0.90) -> Low Risk (0/100).",
    },
}

DIAGNOSIS_FILTER_DEFAULTS = {
    "diag_payment": ["COD", "Prepaid"],
    "diag_tiers": [1, 2, 3, 4],
    "diag_categories": ["Fashion", "Home", "Beauty", "Electronics"],
    "diag_vintage": ["First-time", "Repeat"],
    "diag_seller": ["New Seller (<30d)", "Established Seller"],
    "diag_risk_tiers": ["Low", "Medium", "High", "Very High"],
}

ROADMAP_FILTER_DEFAULTS = {
    "rice_phases": ["V2 (60-90d)", "V3 (6mo)", "V4 (9mo)", "V4 (12mo)", "V5 (18mo)"],
    "rice_min_conf": 0.50,
}


# ---------------------------------------------------------------------------
# Data Bootstrapping & Query Layer (Preserved Signatures)
# ---------------------------------------------------------------------------
def ensure_data() -> None:
    """Bootstraps the data layer: generates CSV + SQLite DB on first run if absent."""
    if not (BASE / "rto_engine.db").exists() or not (BASE / "meesho_orders.csv").exists():
        import generate_data

        generate_data.main()


def named_queries() -> dict[str, str]:
    """Parses queries.sql into a dict keyed by `-- name: <tag>`."""
    text = (BASE / "queries.sql").read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for block in text.split(";"):
        m = re.search(r"--\s*name:\s*(\w+)", block)
        if m:
            out[m.group(1)] = block.strip()
    return out


@st.cache_data(show_spinner=False)
def q(name: str) -> pd.DataFrame:
    """Executes a named query from queries.sql against rto_engine.db."""
    ensure_data()
    sql_map = named_queries()
    if name not in sql_map:
        raise KeyError(f"Unknown SQL query name: {name}")
    with sqlite3.connect(BASE / "rto_engine.db") as con:
        return pd.read_sql_query(sql_map[name], con)


@st.cache_data(show_spinner=False)
def load_meta() -> dict[str, float]:
    """Loads scaling and economic metadata from the meta table."""
    ensure_data()
    with sqlite3.connect(BASE / "rto_engine.db") as con:
        df = pd.read_sql_query("SELECT key, value FROM meta", con)
    return {str(row["key"]): float(row["value"]) for _, row in df.iterrows()}


@st.cache_data(show_spinner=False)
def load_enriched_orders() -> pd.DataFrame:
    """Loads orders joined with scorecard outputs for interactive slice diagnosis."""
    ensure_data()
    sql = """
    SELECT
        o.order_id,
        o.order_date,
        o.customer_id,
        o.first_order,
        o.city_tier,
        o.category,
        o.payment_mode,
        o.order_value,
        o.pin_deliverability,
        o.seller_new,
        o.rto,
        o.rto_loss,
        s.risk_prob,
        s.risk_score,
        s.risk_tier,
        s.intervention
    FROM orders o
    JOIN scores s USING (order_id)
    """
    with sqlite3.connect(BASE / "rto_engine.db") as con:
        df = pd.read_sql_query(sql, con)
    df["buyer_vintage"] = np.where(df["first_order"] == 1, "First-time", "Repeat")
    df["seller_tenure"] = np.where(
        df["seller_new"] == 1, "New Seller (<30d)", "Established Seller"
    )
    df["city_tier_label"] = "Tier " + df["city_tier"].astype(str)
    df["is_focus_segment"] = (
        (df["first_order"] == 1)
        & (df["city_tier"] >= 3)
        & (df["category"] == "Fashion")
        & (df["payment_mode"] == "COD")
    )
    return df


# ---------------------------------------------------------------------------
# Pure Analytical & Transformation Helpers (Testable Without UI)
# ---------------------------------------------------------------------------
def filter_diagnosis_orders(
    df: pd.DataFrame,
    payment_modes: list[str],
    city_tiers: list[int],
    categories: list[str],
    buyer_vintage: list[str],
    seller_tenure: list[str],
    risk_tiers: list[str],
) -> pd.DataFrame:
    """Filters the enriched orders DataFrame across user-selected slice dimensions."""
    if (
        not payment_modes
        or not city_tiers
        or not categories
        or not buyer_vintage
        or not seller_tenure
        or not risk_tiers
    ):
        return df.iloc[0:0].copy()

    mask = (
        df["payment_mode"].isin(payment_modes)
        & df["city_tier"].isin(city_tiers)
        & df["category"].isin(categories)
        & df["buyer_vintage"].isin(buyer_vintage)
        & df["seller_tenure"].isin(seller_tenure)
        & df["risk_tier"].isin(risk_tiers)
    )
    return df.loc[mask].copy()


def compute_filtered_diagnosis_summary(
    filtered_df: pd.DataFrame,
    total_orders: int = 25_000,
    total_loss_rs: float = 381_000.0,
    scale: float = 3200.0,
) -> dict[str, Any]:
    """Computes sample and scaled metrics for any filtered slice of orders."""
    n = int(len(filtered_df))
    if n == 0:
        return {
            "sample_orders": 0,
            "scaled_orders_m": 0.0,
            "order_share_pct": 0.0,
            "rto_orders": 0,
            "rto_rate_pct": 0.0,
            "mean_risk_score": 0.0,
            "loss_cr_month": 0.0,
            "loss_share_pct": 0.0,
            "avg_aov_rs": 0.0,
            "avg_pin": 0.0,
            "severity_label": "EMPTY SLICE (0 ORDERS)",
            "severity_badge": "rto-badge--slate",
        }

    rto_orders = int(filtered_df["rto"].sum())
    rto_rate_pct = float(filtered_df["rto"].mean() * 100.0)
    slice_loss_rs = float(filtered_df["rto_loss"].sum())
    loss_cr_month = float(slice_loss_rs * scale / 1e7)
    loss_share_pct = float(
        (slice_loss_rs / total_loss_rs * 100.0) if total_loss_rs > 0 else 0.0
    )
    order_share_pct = float((n / total_orders * 100.0) if total_orders > 0 else 0.0)
    scaled_orders_m = float(n * scale / 1e6)

    if rto_rate_pct >= 25.0:
        sev_label = "CRITICAL RTO CONCENTRATION (>=25%)"
        sev_badge = "rto-badge--red"
    elif rto_rate_pct >= 15.0:
        sev_label = "ELEVATED RTO RISK (15–25%)"
        sev_badge = "rto-badge--amber"
    else:
        sev_label = "CONTAINED RTO RISK (<15%)"
        sev_badge = "rto-badge--emerald"

    return {
        "sample_orders": n,
        "scaled_orders_m": round(scaled_orders_m, 2),
        "order_share_pct": round(order_share_pct, 1),
        "rto_orders": rto_orders,
        "rto_rate_pct": round(rto_rate_pct, 1),
        "mean_risk_score": round(float(filtered_df["risk_score"].mean()), 1),
        "loss_cr_month": round(loss_cr_month, 1),
        "loss_share_pct": round(loss_share_pct, 1),
        "avg_aov_rs": round(float(filtered_df["order_value"].mean()), 0),
        "avg_pin": round(float(filtered_df["pin_deliverability"].mean()), 2),
        "severity_label": sev_label,
        "severity_badge": sev_badge,
    }


def compute_scorecard_contributions(
    city_tier: int,
    category: str,
    aov: float,
    first_order: bool,
    cod: bool,
    seller_new: bool,
    pin: float,
) -> pd.DataFrame:
    """Computes transparent feature-by-feature log-odds contributions from W."""
    f_cod = float(cod)
    f_first = float(first_order)
    f_tier3 = float(city_tier >= 3)
    f_fashion = float(category == "Fashion")
    f_band = float(300 <= aov <= 900)
    f_seller = float(seller_new)
    f_pin_risk = round(0.60 - float(pin), 4)

    rows = [
        {
            "Signal / Term": "Base Intercept",
            "Input State": "Constant baseline",
            "Weight (W)": W["base"],
            "Active Multiplier": 1.0,
            "Log-Odds Impact": W["base"],
            "Role": "Baseline",
        },
        {
            "Signal / Term": "Payment Intent (COD)",
            "Input State": "COD" if cod else "Prepaid (UPI)",
            "Weight (W)": W["cod"],
            "Active Multiplier": f_cod,
            "Log-Odds Impact": W["cod"] * f_cod,
            "Role": "Risk Driver" if f_cod > 0 else "Inactive",
        },
        {
            "Signal / Term": "Buyer Vintage (First-Time)",
            "Input State": "First-time" if first_order else "Repeat",
            "Weight (W)": W["first"],
            "Active Multiplier": f_first,
            "Log-Odds Impact": W["first"] * f_first,
            "Role": "Risk Driver" if f_first > 0 else "Inactive",
        },
        {
            "Signal / Term": "COD x First-Time Interaction",
            "Input State": "Active" if (cod and first_order) else "Inactive",
            "Weight (W)": W["cod_first"],
            "Active Multiplier": f_cod * f_first,
            "Log-Odds Impact": W["cod_first"] * (f_cod * f_first),
            "Role": "Interaction Driver" if (f_cod * f_first) > 0 else "Inactive",
        },
        {
            "Signal / Term": "Geography (Tier 3+)",
            "Input State": f"Tier {city_tier}",
            "Weight (W)": W["tier3"],
            "Active Multiplier": f_tier3,
            "Log-Odds Impact": W["tier3"] * f_tier3,
            "Role": "Risk Driver" if f_tier3 > 0 else "Inactive",
        },
        {
            "Signal / Term": "First-Time x Tier 3+ Interaction",
            "Input State": "Active" if (first_order and city_tier >= 3) else "Inactive",
            "Weight (W)": W["first_tier3"],
            "Active Multiplier": f_first * f_tier3,
            "Log-Odds Impact": W["first_tier3"] * (f_first * f_tier3),
            "Role": "Interaction Driver" if (f_first * f_tier3) > 0 else "Inactive",
        },
        {
            "Signal / Term": "Category (Fashion)",
            "Input State": category,
            "Weight (W)": W["fashion"],
            "Active Multiplier": f_fashion,
            "Log-Odds Impact": W["fashion"] * f_fashion,
            "Role": "Risk Driver" if f_fashion > 0 else "Inactive",
        },
        {
            "Signal / Term": "Impulse AOV Band (Rs.300–Rs.900)",
            "Input State": f"Rs.{int(aov)} ({'In band' if f_band else 'Outside band'})",
            "Weight (W)": W["band"],
            "Active Multiplier": f_band,
            "Log-Odds Impact": W["band"] * f_band,
            "Role": "Risk Driver" if f_band > 0 else "Inactive",
        },
        {
            "Signal / Term": "Seller Tenure (<30 days)",
            "Input State": "New / Unverified" if seller_new else "Established (>=30d)",
            "Weight (W)": W["seller_new"],
            "Active Multiplier": f_seller,
            "Log-Odds Impact": W["seller_new"] * f_seller,
            "Role": "Risk Driver" if f_seller > 0 else "Inactive",
        },
        {
            "Signal / Term": "PIN Deliverability Gap (0.60 - PIN)",
            "Input State": f"PIN score {pin:.2f} (gap {f_pin_risk:+.2f})",
            "Weight (W)": W["pin"],
            "Active Multiplier": f_pin_risk,
            "Log-Odds Impact": round(W["pin"] * f_pin_risk, 4),
            "Role": "Risk Driver" if f_pin_risk > 0 else "Risk Mitigator",
        },
    ]
    return pd.DataFrame(rows)


def compute_simulator_economics(profile_res: dict[str, Any], aov: float) -> dict[str, float]:
    """Computes per-order unit economics from score_profile() outputs."""
    base_prob = float(profile_res["risk_prob"])
    post_prob = float(profile_res["post_prob"])
    base_expected_loss_rs = base_prob * RTO_COST
    post_expected_loss_rs = post_prob * RTO_COST
    loss_avoided_rs = float(profile_res["loss_avoided"])
    discount_cost_rs = float(profile_res["discount_cost"])
    surcharge_rev_rs = float(profile_res["surcharge"])
    lost_frac = float(profile_res["lost_frac"])
    conv_cost_rs = lost_frac * TAKE_RATE * float(aov)
    net_unit_value_rs = (
        loss_avoided_rs + surcharge_rev_rs - discount_cost_rs - conv_cost_rs
    )
    rel_rto_cut_pct = (
        ((base_prob - post_prob) / base_prob * 100.0) if base_prob > 0 else 0.0
    )
    return {
        "base_expected_loss_rs": round(base_expected_loss_rs, 2),
        "post_expected_loss_rs": round(post_expected_loss_rs, 2),
        "loss_avoided_rs": round(loss_avoided_rs, 2),
        "discount_cost_rs": round(discount_cost_rs, 2),
        "surcharge_rev_rs": round(surcharge_rev_rs, 2),
        "conv_drop_pct": round(lost_frac * 100.0, 2),
        "conv_cost_rs": round(conv_cost_rs, 2),
        "net_unit_value_rs": round(net_unit_value_rs, 2),
        "rel_rto_cut_pct": round(rel_rto_cut_pct, 1),
    }


def enrich_tier_detail(
    tier_df: pd.DataFrame, total_orders: int = 25_000, scale: float = 3200.0
) -> pd.DataFrame:
    """Orders tier_detail from Low -> Very High and adds net benefit and share columns."""
    out = tier_df.copy()
    out["tier_order"] = out["tier"].map({t: i for i, t in enumerate(TIER_ORDER)})
    out = out.sort_values("tier_order").drop(columns=["tier_order"]).reset_index(drop=True)
    out["scaled_orders_m"] = (out["orders"] * scale / 1e6).round(2)
    out["order_share_pct"] = (out["orders"] / total_orders * 100.0).round(1)
    out["base_rto_pct"] = (out["base_rto"] * 100.0).round(1)
    out["post_rto_pct"] = (out["post_rto"] * 100.0).round(1)
    out["rto_cut_pp"] = ((out["base_rto"] - out["post_rto"]) * 100.0).round(1)
    out["net_savings_cr"] = (
        out["avoided_cr"] + out["surcharge_cr"] - out["reward_cr"] - out["conv_cost_cr"]
    ).round(1)
    out["intervention_policy"] = out["tier"].map(INTERVENTION)
    return out


def filter_rice_roadmap(
    rice_df: pd.DataFrame, phases: list[str], min_confidence: float
) -> pd.DataFrame:
    """Filters RICE initiatives by phase and confidence threshold, adding priority bands."""
    if not phases:
        return rice_df.iloc[0:0].copy()
    out = rice_df[
        rice_df["phase"].isin(phases) & (rice_df["confidence"] >= min_confidence)
    ].copy()
    if out.empty:
        return out
    out = out.sort_values("rice", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out["confidence_pct"] = (out["confidence"] * 100.0).round(0).astype(int)
    out["priority_band"] = np.select(
        [out["rice"] >= 12.0, out["rice"] >= 8.0, out["rice"] >= 5.0],
        [
            "P0 · Immediate Unlock",
            "P1 · Near-Term Scale",
            "P2 · Medium-Term Multiplier",
        ],
        default="P3 · Long-Term Bet",
    )
    return out


# ---------------------------------------------------------------------------
# Session State Management & Reset Callbacks
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    """Initializes all interactive widget states if not already present."""
    for k, v in SIMULATOR_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if "sim_preset_selector" not in st.session_state:
        st.session_state["sim_preset_selector"] = "Focus Segment (High Risk Default)"

    for k, v in DIAGNOSIS_FILTER_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = list(v) if isinstance(v, list) else v

    for k, v in ROADMAP_FILTER_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = list(v) if isinstance(v, list) else v


def apply_simulator_preset() -> None:
    """Callback when user chooses a preset scenario in the Checkout Simulator."""
    preset_name = st.session_state.get(
        "sim_preset_selector", "Focus Segment (High Risk Default)"
    )
    preset = SIMULATOR_PRESETS.get(
        preset_name, SIMULATOR_PRESETS["Focus Segment (High Risk Default)"]
    )
    for k in SIMULATOR_DEFAULTS:
        st.session_state[k] = preset[k]


def reset_simulator_controls() -> None:
    """Resets Checkout Simulator inputs to default high-risk focus profile."""
    st.session_state["sim_preset_selector"] = "Focus Segment (High Risk Default)"
    for k, v in SIMULATOR_DEFAULTS.items():
        st.session_state[k] = v


def reset_diagnosis_filters() -> None:
    """Resets Risk Diagnosis segment filters to full-catalog scope."""
    for k, v in DIAGNOSIS_FILTER_DEFAULTS.items():
        st.session_state[k] = list(v) if isinstance(v, list) else v


def select_focus_segment_only() -> None:
    """Sets Risk Diagnosis filters to isolate the Tier 3+ First-time Fashion COD segment."""
    st.session_state["diag_payment"] = ["COD"]
    st.session_state["diag_tiers"] = [3, 4]
    st.session_state["diag_categories"] = ["Fashion"]
    st.session_state["diag_vintage"] = ["First-time"]
    st.session_state["diag_seller"] = ["New Seller (<30d)", "Established Seller"]
    st.session_state["diag_risk_tiers"] = ["Low", "Medium", "High", "Very High"]


def reset_roadmap_filters() -> None:
    """Resets Prioritization roadmap filters to all phases."""
    for k, v in ROADMAP_FILTER_DEFAULTS.items():
        st.session_state[k] = list(v) if isinstance(v, list) else v


def reset_all_workspace_controls() -> None:
    """Global reset across Simulator, Diagnosis, and Roadmap controls."""
    reset_simulator_controls()
    reset_diagnosis_filters()
    reset_roadmap_filters()


# ---------------------------------------------------------------------------
# Minimal, Maintainable Semantic CSS & Escaped Badge Renderer
# ---------------------------------------------------------------------------
def inject_workspace_styles() -> None:
    """Injects minimal, semantic CSS without fragile auto-generated class selectors."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
            max-width: 1400px;
        }
        [data-testid="stMetricValue"] {
            font-size: clamp(1.1rem, 1.4vw, 1.7rem);
            white-space: normal;
            overflow-wrap: anywhere;
        }
        [data-testid="stMetricLabel"] p,
        [data-testid="stMetricDelta"] {
            white-space: normal;
            overflow-wrap: anywhere;
        }
        .rto-header-card {
            background-color: #131C2E;
            border: 1px solid #1E293B;
            border-left: 4px solid #10B981;
            border-radius: 8px;
            padding: 1.1rem 1.35rem;
            margin-bottom: 1rem;
        }
        .rto-header-title {
            font-size: 1.55rem;
            font-weight: 700;
            color: #F8FAFC;
            margin: 0 0 0.35rem 0;
            letter-spacing: -0.01em;
        }
        .rto-header-sub {
            font-size: 0.92rem;
            color: #CBD5E1;
            margin: 0;
            line-height: 1.5;
        }
        .rto-badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 0.65rem;
        }
        .rto-badge {
            display: inline-block;
            font-size: 0.73rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            padding: 0.2rem 0.6rem;
            border-radius: 4px;
            text-transform: uppercase;
        }
        .rto-badge--emerald {
            background-color: #064E3B;
            color: #6EE7B7;
            border: 1px solid #059669;
        }
        .rto-badge--amber {
            background-color: #451A03;
            color: #FCD34D;
            border: 1px solid #D97706;
        }
        .rto-badge--red {
            background-color: #450A0A;
            color: #FCA5A5;
            border: 1px solid #DC2626;
        }
        .rto-badge--slate {
            background-color: #1E293B;
            color: #E2E8F0;
            border: 1px solid #475569;
        }
        .rto-callout {
            background-color: #111827;
            border: 1px solid #1E293B;
            border-radius: 6px;
            padding: 0.85rem 1.1rem;
            margin: 0.5rem 0 0.9rem 0;
            color: #E2E8F0;
            font-size: 0.9rem;
            line-height: 1.5;
        }
        .rto-callout--emerald {
            border-left: 4px solid #10B981;
        }
        .rto-callout--amber {
            border-left: 4px solid #F59E0B;
        }
        .rto-callout--red {
            border-left: 4px solid #EF4444;
        }
        .rto-step-label {
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #10B981;
            margin-bottom: 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_badge(text: str, variant: str = "rto-badge--slate") -> str:
    """Returns escaped HTML for a semantic status badge with explicit text label."""
    safe_text = html.escape(str(text))
    safe_variant = (
        variant
        if variant
        in {
            "rto-badge--emerald",
            "rto-badge--amber",
            "rto-badge--red",
            "rto-badge--slate",
        }
        else "rto-badge--slate"
    )
    return f'<span class="rto-badge {safe_variant}">{safe_text}</span>'


def render_callout(
    title: str,
    body: str,
    variant: str = "rto-callout--emerald",
    badge_html: str = "",
) -> None:
    """Renders an escaped semantic callout panel."""
    safe_title = html.escape(str(title))
    safe_body = html.escape(str(body))
    header_part = (
        f"{badge_html} <strong>{safe_title}</strong>"
        if badge_html
        else f"<strong>{safe_title}</strong>"
    )
    st.markdown(
        f'<div class="rto-callout {variant}">{header_part}<br/><span>{safe_body}</span></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Vega-Lite Chart Specifications (Zero Extra Dependencies, Native Streamlit)
# ---------------------------------------------------------------------------
def render_dimension_chart(
    dim_df: pd.DataFrame, value_col: str, title: str, color_hex: str = "#10B981"
) -> None:
    """Renders a readable horizontal bar chart for segment comparison."""
    chart_df = dim_df.copy()
    chart_df["rto_rate_pct"] = (chart_df["rto_rate"] * 100.0).round(1)
    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": {
            "text": title,
            "color": "#F1F5F9",
            "fontSize": 13,
            "anchor": "start",
        },
        "background": "transparent",
        "height": 195,
        "mark": {"type": "bar", "cornerRadiusEnd": 4, "color": color_hex},
        "encoding": {
            "y": {
                "field": "slice",
                "type": "nominal",
                "sort": "-x",
                "title": None,
                "axis": {"labelColor": "#E2E8F0", "labelFontSize": 12},
            },
            "x": {
                "field": value_col,
                "type": "quantitative",
                "title": (
                    "RTO Rate (%)"
                    if value_col == "rto_rate_pct"
                    else "Share of Total RTO Loss (%)"
                ),
                "axis": {
                    "labelColor": "#94A3B8",
                    "titleColor": "#94A3B8",
                    "gridColor": "#1E293B",
                },
            },
            "tooltip": [
                {"field": "slice", "type": "nominal", "title": "Segment"},
                {
                    "field": "orders",
                    "type": "quantitative",
                    "title": "Sample Orders (n)",
                    "format": ",",
                },
                {
                    "field": "rto_rate_pct",
                    "type": "quantitative",
                    "title": "RTO Rate (%)",
                    "format": ".1f",
                },
                {
                    "field": "loss_share_pct",
                    "type": "quantitative",
                    "title": "Loss Share (%)",
                    "format": ".1f",
                },
            ],
        },
        "config": {"view": {"stroke": "transparent"}},
    }
    st.vega_lite_chart(chart_df, spec, use_container_width=True)


def render_pareto_chart(pareto_df: pd.DataFrame) -> None:
    """Renders a layered bar + cumulative line Pareto chart across risk-score deciles."""
    chart_df = pareto_df.copy()
    chart_df["rto_rate_pct"] = (chart_df["rto_rate"] * 100.0).round(1)
    chart_df["decile_label"] = "D" + chart_df["decile"].astype(str)
    chart_df["risk_zone"] = np.where(
        chart_df["decile"] >= 8,
        "Top 3 Deciles (70.9% of Loss)",
        "Deciles 1–7 (29.1% of Loss)",
    )
    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "background": "transparent",
        "height": 290,
        "layer": [
            {
                "mark": {
                    "type": "bar",
                    "cornerRadiusTopLeft": 3,
                    "cornerRadiusTopRight": 3,
                },
                "encoding": {
                    "x": {
                        "field": "decile",
                        "type": "ordinal",
                        "title": "Risk Score Decile (1 = Lowest Risk -> 10 = Highest Risk)",
                        "axis": {
                            "labelColor": "#E2E8F0",
                            "titleColor": "#94A3B8",
                            "labelAngle": 0,
                        },
                    },
                    "y": {
                        "field": "loss_share_pct",
                        "type": "quantitative",
                        "title": "Decile Share of Total RTO Loss (%)",
                        "axis": {
                            "labelColor": "#94A3B8",
                            "titleColor": "#94A3B8",
                            "gridColor": "#1E293B",
                        },
                    },
                    "color": {
                        "field": "risk_zone",
                        "type": "nominal",
                        "title": "Concentration Zone",
                        "scale": {
                            "domain": [
                                "Top 3 Deciles (70.9% of Loss)",
                                "Deciles 1–7 (29.1% of Loss)",
                            ],
                            "range": ["#F59E0B", "#334155"],
                        },
                        "legend": {
                            "labelColor": "#E2E8F0",
                            "titleColor": "#94A3B8",
                            "orient": "top",
                        },
                    },
                    "tooltip": [
                        {"field": "decile", "type": "ordinal", "title": "Risk Decile"},
                        {
                            "field": "orders",
                            "type": "quantitative",
                            "title": "Sample Orders (n)",
                            "format": ",",
                        },
                        {
                            "field": "rto_rate_pct",
                            "type": "quantitative",
                            "title": "Decile RTO Rate (%)",
                            "format": ".1f",
                        },
                        {
                            "field": "loss_share_pct",
                            "type": "quantitative",
                            "title": "Decile Loss Share (%)",
                            "format": ".1f",
                        },
                        {
                            "field": "cum_loss_share_pct",
                            "type": "quantitative",
                            "title": "Cumulative Loss Share (%)",
                            "format": ".1f",
                        },
                    ],
                },
            },
            {
                "mark": {
                    "type": "line",
                    "color": "#10B981",
                    "strokeWidth": 2.5,
                    "point": {"filled": True, "color": "#10B981", "size": 55},
                },
                "encoding": {
                    "x": {"field": "decile", "type": "ordinal"},
                    "y": {
                        "field": "cum_loss_share_pct",
                        "type": "quantitative",
                        "title": "Cumulative Loss Share (%)",
                        "axis": {"labelColor": "#10B981", "titleColor": "#10B981"},
                    },
                },
            },
        ],
        "resolve": {"scale": {"y": "independent"}},
        "config": {"view": {"stroke": "transparent"}},
    }
    st.vega_lite_chart(chart_df, spec, use_container_width=True)


def render_waterfall_chart(wf_df: pd.DataFrame) -> None:
    """Renders the unit-economics bridge chart from q('waterfall')."""
    chart_df = wf_df.copy()
    role_map = {
        "Baseline RTO loss": "Baseline burden (reference)",
        "RTO loss avoided": "Cost Reduction / Revenue Gain",
        "COD surcharge revenue": "Cost Reduction / Revenue Gain",
        "UPI discount cost": "Incentive & Friction Cost",
        "Conversion cost": "Incentive & Friction Cost",
        "Net monthly savings": "Modeled net benefit (total)",
    }
    chart_df["category_role"] = chart_df["component"].map(role_map)
    chart_df["sort_order"] = np.arange(len(chart_df))
    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "background": "transparent",
        "height": 290,
        "layer": [
            {
                "mark": {"type": "bar", "cornerRadiusEnd": 4},
                "encoding": {
                    "y": {
                        "field": "component",
                        "type": "nominal",
                        "sort": {"field": "sort_order", "order": "ascending"},
                        "title": None,
                        "axis": {
                            "labelColor": "#E2E8F0",
                            "labelFontSize": 12,
                            "labelLimit": 220,
                        },
                    },
                    "x": {
                        "field": "value_cr",
                        "type": "quantitative",
                        "title": "Monthly Financial Impact (Rs. Cr / month, scaled to 80M orders)",
                        "axis": {
                            "labelColor": "#94A3B8",
                            "titleColor": "#94A3B8",
                            "gridColor": "#1E293B",
                        },
                    },
                    "color": {
                        "field": "category_role",
                        "type": "nominal",
                        "title": "Bridge Role",
                        "scale": {
                            "domain": [
                                "Baseline burden (reference)",
                                "Cost Reduction / Revenue Gain",
                                "Incentive & Friction Cost",
                                "Modeled net benefit (total)",
                            ],
                            "range": ["#64748B", "#10B981", "#F59E0B", "#059669"],
                        },
                        "legend": {
                            "labelColor": "#E2E8F0",
                            "titleColor": "#94A3B8",
                            "orient": "top",
                        },
                    },
                    "tooltip": [
                        {
                            "field": "component",
                            "type": "nominal",
                            "title": "Bridge Component",
                        },
                        {
                            "field": "value_cr",
                            "type": "quantitative",
                            "title": "Impact (Rs. Cr/mo)",
                            "format": "+.1f",
                        },
                        {
                            "field": "category_role",
                            "type": "nominal",
                            "title": "Classification",
                        },
                    ],
                },
            },
            {
                "mark": {"type": "rule", "color": "#94A3B8", "strokeDash": [4, 4]},
                "encoding": {"x": {"datum": 0}},
            },
        ],
        "config": {"view": {"stroke": "transparent"}},
    }
    st.vega_lite_chart(chart_df, spec, use_container_width=True)


def render_rice_chart(rice_df: pd.DataFrame) -> None:
    """Renders a horizontal bar chart of RICE-ranked product initiatives."""
    chart_df = rice_df.copy()
    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "background": "transparent",
        "height": 290,
        "mark": {"type": "bar", "cornerRadiusEnd": 4},
        "encoding": {
            "y": {
                "field": "initiative",
                "type": "nominal",
                "sort": "-x",
                "title": None,
                "axis": {
                    "labelColor": "#E2E8F0",
                    "labelFontSize": 12,
                    "labelLimit": 230,
                },
            },
            "x": {
                "field": "rice",
                "type": "quantitative",
                "title": "RICE Score = (Reach x Impact x Confidence) / Effort",
                "axis": {
                    "labelColor": "#94A3B8",
                    "titleColor": "#94A3B8",
                    "gridColor": "#1E293B",
                },
            },
            "color": {
                "field": "phase",
                "type": "nominal",
                "title": "Rollout Horizon",
                "scale": {
                    "domain": [
                        "V2 (60-90d)",
                        "V3 (6mo)",
                        "V4 (9mo)",
                        "V4 (12mo)",
                        "V5 (18mo)",
                    ],
                    "range": ["#10B981", "#34D399", "#F59E0B", "#FBBF24", "#64748B"],
                },
                "legend": {
                    "labelColor": "#E2E8F0",
                    "titleColor": "#94A3B8",
                    "orient": "top",
                },
            },
            "tooltip": [
                {"field": "initiative", "type": "nominal", "title": "Initiative"},
                {"field": "phase", "type": "nominal", "title": "Phase"},
                {
                    "field": "rice",
                    "type": "quantitative",
                    "title": "RICE Score",
                    "format": ".1f",
                },
                {
                    "field": "reach",
                    "type": "quantitative",
                    "title": "Reach (10M/mo)",
                    "format": ".1f",
                },
                {
                    "field": "impact",
                    "type": "quantitative",
                    "title": "Impact Score",
                    "format": ".1f",
                },
                {
                    "field": "confidence_pct",
                    "type": "quantitative",
                    "title": "Confidence (%)",
                },
                {
                    "field": "effort",
                    "type": "quantitative",
                    "title": "Effort (person-months)",
                    "format": ".1f",
                },
            ],
        },
        "config": {"view": {"stroke": "transparent"}},
    }
    st.vega_lite_chart(chart_df, spec, use_container_width=True)


# ---------------------------------------------------------------------------
# Methodology & Assumptions Expander (Reusable Fact-Integrity Component)
# ---------------------------------------------------------------------------
def render_methodology_expander(expanded: bool = False) -> None:
    """Renders the methodology, synthetic data parameters, and scorecard formulas."""
    with st.expander(
        "Methodology, Scorecard Specification & Economic Assumptions (Case Study Reference)",
        expanded=expanded,
    ):
        m_col1, m_col2, m_col3 = st.columns(3, gap="medium")
        with m_col1:
            st.markdown("#### 1. Synthetic Cohort & Scaling")
            st.markdown(
                "- **Case Study Scope:** Independent product & risk-economics teardown built on a deterministic synthetic sample (`n = 25,000` orders, `SEED = 42`).\n"
                "- **Monthly Book Scaling:** Each sampled order scales by **`3,200x`** (`80,000,000 / 25,000`) to model an **80M monthly order book**.\n"
                "- **Direct Unit Loss (`RTO_COST`):** **`Rs.100 / RTO order`** (`Rs.60` forward shipping + `Rs.40` reverse logistics & handling).\n"
                "- **Platform Take-Rate (`TAKE_RATE`):** **`10.0%` of order GMV**, used to price the opportunity cost of checkout abandonment."
            )
        with m_col2:
            st.markdown("#### 2. Specified Logistic Scorecard (`W`)")
            st.markdown(
                "- **Scorecard vs. Trained ML:** Order scoring uses a **specified logistic risk scorecard** (`p = sigmoid(z)`) with calibrated domain weights, **not** an empirical ML model trained on live production logs.\n"
                "- **Logit Weights (`W`):** Base `-4.90`, COD `+0.70`, First-Time `+0.75`, Tier 3+ `+0.25`, First-Time x Tier 3+ `+0.60`, COD x First-Time `+0.55`, Fashion `+0.30`, Impulse Band (`Rs.300–Rs.900`) `+0.40`, New Seller (`<30d`) `+0.50`, PIN Risk `2.85 x (0.60 - PIN)`.\n"
                "- **Risk Tier Cutoffs (`TIER_CUTS`):** `Low < 0.15`, `Medium 0.15–0.35`, `High 0.35–0.60`, `Very High >= 0.60`."
            )
        with m_col3:
            st.markdown("#### 3. Modeled Intervention Mechanics")
            high_stay_pct = (1.0 - TAKEUP["High"] - DROP["High"]) * 100.0
            st.markdown(
                f"- **Low (`<15%`):** Standard 1-click COD (`0%` added friction).\n"
                f"- **Medium (`15–35%`):** Post-order WhatsApp address/slot confirmation (`-{int(CONFIRM_RTO_CUT*100)}%` relative RTO cut, `0%` drop).\n"
                f"- **High (`35–60%`):** `Rs.{int(UPI_DISCOUNT)}` UPI instant discount (`{int(TAKEUP['High']*100)}%` prepaid conversion) or `Rs.{int(COD_SURCHARGE)}` COD surcharge (`{high_stay_pct:.1f}%` stay with `-{int(SURCHARGE_RTO_CUT*100)}%` RTO cut; `{DROP['High']*100:.1f}%` drop).\n"
                f"- **Very High (`>=60%`):** Prepaid-only checkout gate with `Rs.{int(UPI_DISCOUNT)}` UPI discount (`{int(TAKEUP['Very High']*100)}%` convert to prepaid, `{int(DROP['Very High']*100)}%` abandon; `0%` remain on COD)."
            )


# ---------------------------------------------------------------------------
# Main Application Entrypoint
# ---------------------------------------------------------------------------
def prepare_waterfall_display(raw: pd.DataFrame) -> pd.DataFrame:
    """Translate the legacy cost-reduction sign to a positive benefit for display."""
    display = raw.copy()
    avoided = display["component"].eq("RTO loss avoided")
    display.loc[avoided, "value_cr"] = display.loc[avoided, "value_cr"].abs()
    return display


def main() -> None:
    st.set_page_config(
        page_title="Meesho COD RTO Decision Workspace",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    ensure_data()
    init_session_state()
    inject_workspace_styles()

    # Load canonical SQL tables
    topline_df = q("topline")
    focus_df = q("focus_split").copy()
    focus_df["focus"] = focus_df["focus"].map({1: "Tier-3+ COD Fashion First-Timer", 0: "Rest of Catalog"})
    dim_df = q("by_dimension")
    pareto_df = q("pareto")
    wf_df = prepare_waterfall_display(q("waterfall"))
    kpi_df = q("savings_kpi")
    tier_raw_df = q("tier_detail")
    rice_raw_df = q("rice")
    meta = load_meta()
    orders_df = load_enriched_orders()

    top = topline_df.iloc[0]
    kpi = kpi_df.iloc[0]
    focus_row = focus_df[focus_df["focus"] == "Tier-3+ COD Fashion First-Timer"].iloc[0]
    rest_row = focus_df[focus_df["focus"] == "Rest of Catalog"].iloc[0]
    tier_df = enrich_tier_detail(
        tier_raw_df,
        total_orders=int(top["orders"]),
        scale=float(meta.get("scale", 3200.0)),
    )

    # -----------------------------------------------------------------------
    # Sidebar: Scenario Context, Guardrails & Global Reset
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### Workspace Context")
        st.markdown(
            f"{render_badge('SYNTHETIC CASE STUDY', 'rto-badge--emerald')} "
            f"{render_badge('MODELED ECONOMICS', 'rto-badge--slate')}",
            unsafe_allow_html=True,
        )
        st.caption(
            "All metrics are computed at runtime from `rto_engine.db` via `queries.sql` "
            "and `risk_engine.py`."
        )

        with st.container(border=True):
            st.markdown("**Denominator & Scale Basis**")
            st.caption(
                f"• **Synthetic Sample:** `{int(top['orders']):,}` orders (`SEED=42`)\n"
                f"• **Monthly Book Scale:** `80,000,000` orders (`3,200x`)\n"
                f"• **Unit RTO Cost:** `Rs.{int(RTO_COST)}` / returned order\n"
                f"• **GMV Take-Rate:** `{TAKE_RATE*100:.0f}%` on order value\n"
                f"• **Conversion Guardrail:** `< 1.50%` order drop"
            )

        with st.container(border=True):
            st.markdown("**Scorecard Risk Tiers**")
            st.caption(
                "• **Low (`<15%`):** Standard COD (`62.7%` of orders)\n"
                "• **Medium (`15–35%`):** WhatsApp Nudge (`23.3%`)\n"
                "• **High (`35–60%`):** UPI Rs.40 / COD +Rs.30 (`13.4%`)\n"
                "• **Very High (`>=60%`):** Prepaid-Only Gate (`0.6%`)"
            )

        st.button(
            "Reset All Workspace Controls",
            on_click=reset_all_workspace_controls,
            use_container_width=True,
            help="Restores default filters and simulator inputs across all tabs.",
        )

    # -----------------------------------------------------------------------
    # Top Header Banner
    # -----------------------------------------------------------------------
    badges_html = (
        render_badge("CASE STUDY: SYNTHETIC ORDERS (n = 25,000 -> 80M/mo)", "rto-badge--emerald")
        + render_badge("SCORECARD: SPECIFIED LOGIT WEIGHTS", "rto-badge--slate")
        + render_badge("INTERVENTIONS: MODELED OUTCOMES", "rto-badge--amber")
    )
    st.markdown(
        f"""
        <div class="rto-header-card">
            <div class="rto-badge-row">{badges_html}</div>
            <div class="rto-header-title">Meesho COD Return-to-Origin (RTO) Risk &amp; Unit-Economics Decision Workspace</div>
            <p class="rto-header-sub">
                Evaluating selective checkout gating and payment-mode nudges to reduce Cash-on-Delivery return losses without breaching platform conversion guardrails.
                Denominator basis: <strong>25,000 synthetic orders</strong> scaled <strong>3,200&times;</strong> to an <strong>80M monthly order book</strong> at <strong>Rs.100 direct loss per RTO</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------
    # Primary Navigation Tabs (5 Required Sections)
    # -----------------------------------------------------------------------
    tab_overview, tab_diagnosis, tab_simulator, tab_economics, tab_roadmap = st.tabs(
        [
            "1 · Overview & Headline KPIs",
            "2 · Risk Diagnosis & Concentration",
            "3 · Checkout Simulator",
            "4 · Unit Economics & Waterfall",
            "5 · RICE Prioritization",
        ]
    )

    # =======================================================================
    # TAB 1: OVERVIEW
    # =======================================================================
    with tab_overview:
        st.subheader("Executive Overview: Framing the COD RTO Decision")
        render_callout(
            "Core Business Question — Checkout Gating vs. Blanket COD Restriction:",
            f"At 80M monthly orders, a {top['blended_rto']*100:.1f}% blended RTO rate creates Rs.{top['loss_cr_month']:.1f} Cr/month in direct forward-and-reverse logistics drag. "
            f"Because {focus_row['loss_share_pct']:.1f}% of that loss (Rs.{focus_row['loss_cr_month']:.1f} Cr/month) originates from just {focus_row['order_share_pct']:.1f}% of orders "
            f"(First-Time x Tier-3+ x Fashion x COD), blanket COD restrictions would needlessly penalize 70.7% of the catalog where RTO averages only {rest_row['rto_rate']*100:.1f}%. "
            f"A risk-tiered checkout engine targets interventions strictly where expected RTO loss exceeds conversion friction.",
            variant="rto-callout--emerald",
            badge_html=render_badge("80/20 STRATEGIC FINDING", "rto-badge--emerald"),
        )

        # Headline KPI Row (Computed live from topline, focus_split, savings_kpi)
        k1, k2, k3, k4 = st.columns(4, gap="medium")
        with k1:
            with st.container(border=True):
                st.markdown(render_badge("BASELINE RATE", "rto-badge--amber"), unsafe_allow_html=True)
                st.metric(
                    label="Blended Platform RTO Rate",
                    value=f"{top['blended_rto']*100:.1f}%",
                    delta=f"COD {top['cod_rto']*100:.1f}% vs Prepaid {top['prepaid_rto']*100:.1f}%",
                    delta_color="off",
                    help="Computed from q('topline') across n = 25,000 synthetic orders (19,632 COD / 5,368 Prepaid).",
                )
                st.caption("Unit: `% of total orders` | Basis: `25,000` sample (`80M/mo`)")

        with k2:
            with st.container(border=True):
                st.markdown(render_badge("DIRECT COST DRAG", "rto-badge--red"), unsafe_allow_html=True)
                st.metric(
                    label="Baseline Direct RTO Loss",
                    value=f"Rs.{top['loss_cr_month']:.1f} Cr/mo",
                    delta=f"Rs.{focus_row['loss_cr_month']:.1f} Cr/mo in focus segment",
                    delta_color="inverse",
                    help="Computed from SUM(rto_loss) * 3,200 / 1e7 at Rs.100 direct logistics loss per RTO.",
                )
                st.caption("Unit: `Rs. Crore / month` | Rate: `Rs.100 / RTO order`")

        with k3:
            with st.container(border=True):
                st.markdown(render_badge("LOSS CONCENTRATION", "rto-badge--red"), unsafe_allow_html=True)
                st.metric(
                    label="Focus-Segment Loss Share",
                    value=f"{focus_row['loss_share_pct']:.1f}%",
                    delta=f"{focus_row['rto_rate']*100:.1f}% RTO vs {rest_row['rto_rate']*100:.1f}% rest of book",
                    delta_color="inverse",
                    help="First-Time x Tier-3+ x Fashion x COD represents 29.3% of orders (7,317 / 25,000) but 65.8% of RTO loss.",
                )
                st.caption(f"Scope: `{int(focus_row['orders']):,}` sample orders (`29.3%` of volume)")

        with k4:
            with st.container(border=True):
                st.markdown(render_badge("MODELED NET UNLOCK", "rto-badge--emerald"), unsafe_allow_html=True)
                st.metric(
                    label="Modeled Net Monthly Savings",
                    value=f"Rs.{kpi['net_savings_cr']:.1f} Cr/mo",
                    delta=f"{kpi['pct_of_baseline']:.1f}% of baseline | {kpi['conversion_drop_pct']:.2f}% conv. drop",
                    delta_color="normal",
                    help="Modeled net savings = RTO avoided (Rs.18.7 Cr) + COD surcharge (Rs.24.3 Cr) - UPI discount (Rs.10.8 Cr) - conversion cost (Rs.2.4 Cr).",
                )
                st.caption(f"Guardrail <1.50%: {'PASS' if kpi['conversion_drop_pct'] < 1.5 else 'BREACH'} | Modeled target Rs.15 Cr/mo: {'MET' if kpi['net_savings_cr'] >= 15 else 'MISSED'}")

        # Decision Architecture & Segment Snapshot
        ov_left, ov_right = st.columns([1.15, 1.0], gap="medium")
        with ov_left:
            with st.container(border=True):
                st.markdown("#### Concentration Split: Focus Segment vs. Rest of Catalog")
                st.caption(
                    "Computed via `q('focus_split')` — comparing `First-Time x Tier-3+ x Fashion x COD` against all other orders."
                )
                focus_disp = focus_df.copy()
                focus_disp["scaled_orders_m"] = (focus_disp["orders"] * 3200.0 / 1e6).round(2)
                focus_disp["rto_rate_pct"] = (focus_disp["rto_rate"] * 100.0).round(1)
                st.dataframe(
                    focus_disp[
                        [
                            "focus",
                            "orders",
                            "scaled_orders_m",
                            "order_share_pct",
                            "rto_rate_pct",
                            "loss_cr_month",
                            "loss_share_pct",
                        ]
                    ],
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "focus": st.column_config.TextColumn("Catalog Cohort"),
                        "orders": st.column_config.NumberColumn("Sample Orders (n)", format="%d"),
                        "scaled_orders_m": st.column_config.NumberColumn("Scaled Volume (M/mo)", format="%.2fM"),
                        "order_share_pct": st.column_config.NumberColumn("Order Share (%)", format="%.1f%%"),
                        "rto_rate_pct": st.column_config.NumberColumn("RTO Rate (%)", format="%.1f%%"),
                        "loss_cr_month": st.column_config.NumberColumn("Direct Loss (Rs. Cr/mo)", format="Rs. %.1f Cr"),
                        "loss_share_pct": st.column_config.ProgressColumn(
                            "Share of Total Loss (%)", format="%.1f%%", min_value=0, max_value=100
                        ),
                    },
                )

        with ov_right:
            with st.container(border=True):
                st.markdown("#### Tiered Intervention Policy Summary")
                st.caption(
                    "How the specified logistic scorecard routes orders across four checkout friction bands."
                )
                policy_summary = tier_df[
                    [
                        "tier",
                        "order_share_pct",
                        "base_rto_pct",
                        "post_rto_pct",
                        "net_savings_cr",
                        "intervention_policy",
                    ]
                ].copy()
                st.dataframe(
                    policy_summary,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "tier": st.column_config.TextColumn("Risk Tier"),
                        "order_share_pct": st.column_config.NumberColumn("Book Share", format="%.1f%%"),
                        "base_rto_pct": st.column_config.NumberColumn("Base RTO", format="%.1f%%"),
                        "post_rto_pct": st.column_config.NumberColumn("Modeled Post RTO", format="%.1f%%"),
                        "net_savings_cr": st.column_config.NumberColumn("Net Benefit", format="Rs. %.1f Cr"),
                        "intervention_policy": st.column_config.TextColumn("Checkout Action"),
                    },
                )

        render_methodology_expander(expanded=False)

    # =======================================================================
    # TAB 2: RISK DIAGNOSIS
    # =======================================================================
    with tab_diagnosis:
        st.subheader("Risk Diagnosis: Loss Concentration & Segment Comparisons")
        st.caption(
            "Inspect canonical platform-wide SQL breakdowns (`by_dimension`, `pareto`) and filter the 25,000-order synthetic book to isolate high-loss cohorts."
        )

        # Interactive Slice Filter Bar with Explicit Scope & Reset
        with st.container(border=True):
            f_hdr_l, f_hdr_m, f_hdr_r = st.columns([2.2, 0.9, 0.9], gap="small")
            with f_hdr_l:
                st.markdown("#### Interactive Cohort Filter (Scope: Custom Slice Explorer)")
                st.caption(
                    "Filter the 25,000 scored synthetic orders to inspect cohort-specific RTO rate and loss concentration. Canonical platform charts below remain anchored to full-book SQL queries."
                )
            with f_hdr_m:
                st.button(
                    "Isolate 80/20 Focus Segment",
                    on_click=select_focus_segment_only,
                    use_container_width=True,
                    help="Filters to COD + Tier 3 & 4 + Fashion + First-time buyer (7,317 sample orders).",
                )
            with f_hdr_r:
                st.button(
                    "Reset Diagnosis Filters",
                    on_click=reset_diagnosis_filters,
                    use_container_width=True,
                    help="Restores all 25,000 orders across every dimension.",
                )

            fc1, fc2, fc3 = st.columns(3, gap="medium")
            with fc1:
                sel_payment = st.multiselect(
                    "Payment Mode",
                    options=["COD", "Prepaid"],
                    key="diag_payment",
                )
                sel_vintage = st.multiselect(
                    "Buyer Vintage",
                    options=["First-time", "Repeat"],
                    key="diag_vintage",
                )
            with fc2:
                sel_tiers = st.multiselect(
                    "City Tier",
                    options=[1, 2, 3, 4],
                    format_func=lambda x: f"Tier {x}",
                    key="diag_tiers",
                )
                sel_seller = st.multiselect(
                    "Seller Tenure",
                    options=["New Seller (<30d)", "Established Seller"],
                    key="diag_seller",
                )
            with fc3:
                sel_cats = st.multiselect(
                    "Product Category",
                    options=["Fashion", "Home", "Beauty", "Electronics"],
                    key="diag_categories",
                )
                sel_risk_tiers = st.multiselect(
                    "Assigned Risk Tier",
                    options=TIER_ORDER,
                    key="diag_risk_tiers",
                )

            filtered_orders = filter_diagnosis_orders(
                orders_df,
                payment_modes=sel_payment,
                city_tiers=sel_tiers,
                categories=sel_cats,
                buyer_vintage=sel_vintage,
                seller_tenure=sel_seller,
                risk_tiers=sel_risk_tiers,
            )
            slice_summary = compute_filtered_diagnosis_summary(
                filtered_orders,
                total_orders=int(orders_df.shape[0]),
                total_loss_rs=float(orders_df["rto_loss"].sum()),
                scale=float(meta.get("scale", 3200.0)),
            )

            if slice_summary["sample_orders"] == 0:
                st.warning(
                    "**Empty Filter Slice (0 matching orders):** At least one filter dimension has no selected values or no synthetic orders match this combination. Click **Reset Diagnosis Filters** above to restore the 25,000-order catalog."
                )
            else:
                scope_text = (
                    f"SCOPE: {slice_summary['sample_orders']:,} / 25,000 SAMPLE ORDERS "
                    f"({slice_summary['order_share_pct']:.1f}%)"
                )
                st.markdown(
                    f"{render_badge(slice_summary['severity_label'], slice_summary['severity_badge'])} "
                    f"{render_badge(scope_text, 'rto-badge--slate')}",
                    unsafe_allow_html=True,
                )
                sc1, sc2, sc3, sc4 = st.columns(4, gap="medium")
                sc1.metric(
                    "Filtered Cohort Volume",
                    f"{slice_summary['scaled_orders_m']:.2f}M / mo",
                    delta=f"{slice_summary['order_share_pct']:.1f}% of 80M book (n={slice_summary['sample_orders']:,})",
                    delta_color="off",
                )
                sc2.metric(
                    "Cohort Observed RTO Rate",
                    f"{slice_summary['rto_rate_pct']:.1f}%",
                    delta=f"{slice_summary['rto_rate_pct'] - top['blended_rto']*100:+.1f}pp vs {top['blended_rto']*100:.1f}% platform blend",
                    delta_color="inverse",
                )
                sc3.metric(
                    "Cohort Direct RTO Loss",
                    f"Rs.{slice_summary['loss_cr_month']:.1f} Cr/mo",
                    delta=f"{slice_summary['loss_share_pct']:.1f}% of Rs.{top['loss_cr_month']:.1f} Cr total loss",
                    delta_color="inverse",
                )
                sc4.metric(
                    "Mean Risk Score & PIN Score",
                    f"{slice_summary['mean_risk_score']:.1f} / 100",
                    delta=f"Avg AOV Rs.{int(slice_summary['avg_aov_rs'])} | Avg PIN {slice_summary['avg_pin']:.2f}",
                    delta_color="off",
                )

        # Canonical Single-Dimension Comparisons (from q('by_dimension'))
        st.markdown("#### Platform Baseline Comparisons by Dimension (`q('by_dimension')`)")
        st.caption(
            "Unfiltered full-book benchmarks across Payment Mode, City Tier, and Product Category (`n = 25,000` sample orders)."
        )
        d_col1, d_col2, d_col3 = st.columns(3, gap="medium")

        pm_slice = dim_df[dim_df["dimension"] == "payment_mode"].copy()
        pm_slice["rto_rate_pct"] = (pm_slice["rto_rate"] * 100.0).round(1)
        ct_slice = dim_df[dim_df["dimension"] == "city_tier"].copy()
        ct_slice["rto_rate_pct"] = (ct_slice["rto_rate"] * 100.0).round(1)
        cat_slice = dim_df[dim_df["dimension"] == "category"].copy()
        cat_slice["rto_rate_pct"] = (cat_slice["rto_rate"] * 100.0).round(1)

        with d_col1:
            with st.container(border=True):
                render_dimension_chart(
                    pm_slice, "rto_rate_pct", "RTO Rate by Payment Mode (%)", "#10B981"
                )
                st.dataframe(
                    pm_slice[["slice", "orders", "rto_rate_pct", "loss_share_pct"]],
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "slice": st.column_config.TextColumn("Payment"),
                        "orders": st.column_config.NumberColumn("Orders (n)", format="%d"),
                        "rto_rate_pct": st.column_config.NumberColumn("RTO (%)", format="%.1f%%"),
                        "loss_share_pct": st.column_config.NumberColumn("Loss Share", format="%.1f%%"),
                    },
                )

        with d_col2:
            with st.container(border=True):
                render_dimension_chart(
                    ct_slice, "rto_rate_pct", "RTO Rate by City Tier (%)", "#F59E0B"
                )
                st.dataframe(
                    ct_slice[["slice", "orders", "rto_rate_pct", "loss_share_pct"]],
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "slice": st.column_config.TextColumn("City Tier"),
                        "orders": st.column_config.NumberColumn("Orders (n)", format="%d"),
                        "rto_rate_pct": st.column_config.NumberColumn("RTO (%)", format="%.1f%%"),
                        "loss_share_pct": st.column_config.NumberColumn("Loss Share", format="%.1f%%"),
                    },
                )

        with d_col3:
            with st.container(border=True):
                render_dimension_chart(
                    cat_slice, "rto_rate_pct", "RTO Rate by Category (%)", "#10B981"
                )
                st.dataframe(
                    cat_slice[["slice", "orders", "rto_rate_pct", "loss_share_pct"]],
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "slice": st.column_config.TextColumn("Category"),
                        "orders": st.column_config.NumberColumn("Orders (n)", format="%d"),
                        "rto_rate_pct": st.column_config.NumberColumn("RTO (%)", format="%.1f%%"),
                        "loss_share_pct": st.column_config.NumberColumn("Loss Share", format="%.1f%%"),
                    },
                )

        # Pareto Risk Decile Analysis (from q('pareto'))
        with st.container(border=True):
            st.markdown("#### Risk-Score Decile Pareto Curve (`q('pareto')`)")
            st.caption(
                "Orders sorted into 10 equal deciles (`2,500` sample orders / `8.0M` monthly orders per decile) by specified scorecard score. "
                "Deciles 8, 9, and 10 (the highest-risk 30% of volume) generate **70.9%** (`17.2% + 22.6% + 31.1%`) of total platform RTO loss."
            )
            p_left, p_right = st.columns([1.35, 1.0], gap="medium")
            with p_left:
                render_pareto_chart(pareto_df)
            with p_right:
                pareto_disp = pareto_df.copy()
                pareto_disp["rto_rate_pct"] = (pareto_disp["rto_rate"] * 100.0).round(1)
                st.dataframe(
                    pareto_disp[
                        ["decile", "orders", "rto_rate_pct", "loss_share_pct", "cum_loss_share_pct"]
                    ],
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "decile": st.column_config.NumberColumn("Decile", format="D%d"),
                        "orders": st.column_config.NumberColumn("Orders (n)", format="%d"),
                        "rto_rate_pct": st.column_config.NumberColumn("RTO Rate", format="%.1f%%"),
                        "loss_share_pct": st.column_config.NumberColumn("Loss Share", format="%.1f%%"),
                        "cum_loss_share_pct": st.column_config.ProgressColumn(
                            "Cumulative Loss", format="%.1f%%", min_value=0, max_value=100
                        ),
                    },
                )

    # =======================================================================
    # TAB 3: CHECKOUT SIMULATOR
    # =======================================================================
    with tab_simulator:
        st.subheader("Live Checkout Simulator: Inputs -> Risk Estimate -> Intervention -> Modeled Economics")
        st.caption(
            "Evaluate how the specified logistic scorecard (`score_profile`) scores an individual checkout session in real time and routes it to a tier-specific intervention."
        )

        # Preset Bar & Reset
        with st.container(border=True):
            pr_col1, pr_col2 = st.columns([3.0, 1.0], gap="medium")
            with pr_col1:
                st.selectbox(
                    "Load Representative Checkout Scenario Preset",
                    options=list(SIMULATOR_PRESETS.keys()),
                    key="sim_preset_selector",
                    on_change=apply_simulator_preset,
                    help="Select a preset buyer profile to populate the 7 checkout signals below, or adjust any control manually.",
                )
            with pr_col2:
                st.write("")
                st.button(
                    "Reset Simulator to Default",
                    on_click=reset_simulator_controls,
                    use_container_width=True,
                    help="Restores the default Tier-3 First-time Fashion COD profile (AOV Rs.550, PIN 0.35).",
                )
            active_preset_desc = SIMULATOR_PRESETS.get(
                st.session_state.get("sim_preset_selector", ""), {}
            ).get("desc", "")
            if active_preset_desc:
                st.caption(f"**Preset reference (manual edits may differ):** {active_preset_desc}")

        # 4-Stage Horizontal Decision Flow
        sim_col_in, sim_col_risk, sim_col_econ = st.columns([1.05, 1.15, 1.15], gap="medium")

        # STAGE 1: INPUTS
        with sim_col_in:
            with st.container(border=True):
                st.markdown(
                    '<div class="rto-step-label">Step 1 · Session &amp; Order Inputs</div>',
                    unsafe_allow_html=True,
                )
                st.caption("Configure the 7 order-time signals passed to `score_profile()`.")

                city_tier = st.selectbox(
                    "City tier",
                    options=[1, 2, 3, 4],
                    format_func=lambda x: f"Tier {x} ({'Metro' if x == 1 else 'Regional' if x == 2 else 'Small Town / Rural'})",
                    key="sim_city_tier",
                )
                category = st.selectbox(
                    "Category",
                    options=["Fashion", "Home", "Beauty", "Electronics"],
                    key="sim_category",
                )
                aov = st.slider(
                    "Order value (Rs)",
                    min_value=120,
                    max_value=3000,
                    step=10,
                    key="sim_aov",
                    help="Orders between Rs.300 and Rs.900 inclusive carry a +0.40 impulse-band log-odds weight.",
                )
                buyer_vintage = st.radio(
                    "Buyer vintage",
                    options=["First-time", "Repeat"],
                    horizontal=True,
                    key="sim_buyer_vintage",
                )
                payment_intent = st.radio(
                    "Payment intent",
                    options=["COD", "Prepaid (UPI)"],
                    horizontal=True,
                    key="sim_payment_intent",
                )
                seller_new = st.checkbox(
                    "New / unverified seller (<30 days)",
                    key="sim_seller_new",
                )
                pin = st.slider(
                    "PIN-code deliverability score",
                    min_value=0.05,
                    max_value=0.95,
                    step=0.05,
                    key="sim_pin",
                    help="Historical delivery success index (0.05 = chronic NDR/RTO cluster, 0.95 = high-reliability PIN).",
                )

        first_order_bool = buyer_vintage == "First-time"
        cod_bool = payment_intent == "COD"

        # Compute exact scorecard output & unit economics
        sim_out = score_profile(
            city_tier=city_tier,
            category=category,
            aov=aov,
            first_order=first_order_bool,
            cod=cod_bool,
            seller_new=seller_new,
            pin=pin,
        )
        sim_econ = compute_simulator_economics(sim_out, aov=float(aov))
        contrib_df = compute_scorecard_contributions(
            city_tier=city_tier,
            category=category,
            aov=aov,
            first_order=first_order_bool,
            cod=cod_bool,
            seller_new=seller_new,
            pin=pin,
        )
        t_meta = TIER_META[sim_out["tier"]]

        # STAGE 2 & 3: RISK ESTIMATE & PROPOSED INTERVENTION
        with sim_col_risk:
            with st.container(border=True):
                st.markdown(
                    '<div class="rto-step-label">Step 2 · Scorecard Risk Estimate</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"{render_badge(t_meta['status_text'], t_meta['badge_class'])} "
                    f"{render_badge('SPECIFIED SCORECARD', 'rto-badge--slate')}",
                    unsafe_allow_html=True,
                )

                r_m1, r_m2 = st.columns(2, gap="small")
                r_m1.metric(
                    "Risk Score (0–100)",
                    f"{sim_out['risk_score']} / 100",
                    delta=f"Tier: {sim_out['tier']}",
                    delta_color="off",
                )
                r_m2.metric(
                    "Predicted Baseline RTO",
                    f"{sim_out['risk_prob']*100:.1f}%",
                    delta=f"Rs.{sim_econ['base_expected_loss_rs']:.1f} expected loss/order",
                    delta_color="inverse",
                )

                st.markdown(
                    '<div class="rto-step-label" style="margin-top:0.6rem;">Step 3 · Proposed Checkout Intervention</div>',
                    unsafe_allow_html=True,
                )
                callout_variant = (
                    "rto-callout--emerald"
                    if sim_out["tier"] == "Low"
                    else "rto-callout--amber"
                    if sim_out["tier"] in ("Medium", "High")
                    else "rto-callout--red"
                )
                render_callout(
                    f"Gate Policy ({t_meta['Gate']}):",
                    sim_out["intervention"],
                    variant=callout_variant,
                    badge_html=render_badge(
                        f"TIER: {sim_out['tier'].upper()}", t_meta["badge_class"]
                    ),
                )

                st.markdown("**Scorecard Log-Odds Decomposition (`z = sum(W * x)`)**")
                st.caption(
                    f"Total Logit `z = {contrib_df['Log-Odds Impact'].sum():+.2f}` -> `sigmoid(z) = {sim_out['risk_prob']*100:.1f}%` RTO probability."
                )
                st.dataframe(
                    contrib_df[["Signal / Term", "Input State", "Log-Odds Impact"]],
                    hide_index=True,
                    use_container_width=True,
                    height=275,
                    column_config={
                        "Signal / Term": st.column_config.TextColumn("Signal"),
                        "Input State": st.column_config.TextColumn("Session Value"),
                        "Log-Odds Impact": st.column_config.NumberColumn(
                            "Log-Odds (z)", format="%+.2f"
                        ),
                    },
                )

        # STAGE 4: MODELED UNIT & COHORT ECONOMICS
        with sim_col_econ:
            with st.container(border=True):
                st.markdown(
                    '<div class="rto-step-label">Step 4 · Modeled Session Economics</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"{render_badge('MODELED PER-ORDER OUTCOME', 'rto-badge--emerald')} "
                    f"{render_badge(f'AOV: RS.{int(aov)}', 'rto-badge--slate')}",
                    unsafe_allow_html=True,
                )

                e_m1, e_m2 = st.columns(2, gap="small")
                e_m1.metric(
                    "Post-Intervention RTO",
                    f"{sim_out['post_prob']*100:.1f}%",
                    delta=f"{(sim_out['post_prob'] - sim_out['risk_prob'])*100:+.1f}pp ({-sim_econ['rel_rto_cut_pct']:.1f}% rel.)",
                    delta_color="inverse",
                )
                e_m2.metric(
                    "Direct RTO Loss Avoided",
                    f"Rs.{sim_econ['loss_avoided_rs']:.2f} / ord",
                    delta=f"Rs.{sim_econ['base_expected_loss_rs']:.1f} -> Rs.{sim_econ['post_expected_loss_rs']:.1f}",
                    delta_color="normal",
                )

                e_m3, e_m4 = st.columns(2, gap="small")
                e_m3.metric(
                    "Expected Incentive / Fee",
                    f"Rs.{sim_econ['surcharge_rev_rs'] - sim_econ['discount_cost_rs']:+.2f} / ord",
                    delta=f"Fee +Rs.{sim_econ['surcharge_rev_rs']:.1f} | Disc -Rs.{sim_econ['discount_cost_rs']:.1f}",
                    delta_color="off",
                )
                e_m4.metric(
                    "Net Expected Unit Benefit",
                    f"Rs.{sim_econ['net_unit_value_rs']:+.2f} / ord",
                    delta=f"Conv. drop {sim_econ['conv_drop_pct']:.1f}% (-Rs.{sim_econ['conv_cost_rs']:.2f})",
                    delta_color="normal" if sim_econ["net_unit_value_rs"] >= 0 else "inverse",
                )

                if sim_econ["conv_drop_pct"] >= 20.0:
                    render_callout(
                        "High Abandonment Gate Active (35.0% Modeled Drop):",
                        f"In the Very High tier (>=60% RTO), COD is blocked. While 35% of sessions abandon (costing Rs.{sim_econ['conv_cost_rs']:.2f}/order in lost 10% take-rate), the 65% who convert via UPI cut RTO to {sim_out['post_prob']*100:.1f}%, avoiding Rs.{sim_econ['loss_avoided_rs']:.2f}/order in logistics loss.",
                        variant="rto-callout--red",
                        badge_html=render_badge("CONVERSION FRICTION WARNING", "rto-badge--red"),
                    )
                elif sim_econ["conv_drop_pct"] > 0:
                    render_callout(
                        "Moderate Checkout Friction (2.5% Modeled Drop):",
                        f"22% of High-risk buyers take the Rs.40 UPI discount, 75.5% stay on COD paying the Rs.30 surcharge, and 2.5% abandon checkout (Rs.{sim_econ['conv_cost_rs']:.2f}/order take-rate cost).",
                        variant="rto-callout--amber",
                        badge_html=render_badge("GUARDRAILED FRICTION", "rto-badge--amber"),
                    )
                else:
                    render_callout(
                        "Zero Checkout Abandonment (0.0% Modeled Drop):",
                        "Low and Medium tiers preserve 100% checkout conversion—Medium uses post-order WhatsApp slot confirmation rather than checkout payment gating.",
                        variant="rto-callout--emerald",
                        badge_html=render_badge("ZERO CONVERSION DROP", "rto-badge--emerald"),
                    )

                # Cohort Context from tier_detail
                cohort_row = tier_df[tier_df["tier"] == sim_out["tier"]].iloc[0]
                st.markdown(f"**Book Context for `{sim_out['tier']}` Tier (`q('tier_detail')`)**")
                st.caption(
                    f"• **Catalog Share:** `{cohort_row['order_share_pct']:.1f}%` (`{int(cohort_row['orders']):,}` sample / `{cohort_row['scaled_orders_m']:.2f}M` monthly orders)\n"
                    f"• **Tier Baseline -> Post RTO:** `{cohort_row['base_rto_pct']:.1f}%` -> `{cohort_row['post_rto_pct']:.1f}%` (`-{cohort_row['rto_cut_pp']:.1f}pp`)\n"
                    f"• **Tier Net Monthly Benefit:** **`Rs.{cohort_row['net_savings_cr']:.1f} Cr/month`** (`Rs.{cohort_row['avoided_cr']:.1f} Cr` RTO avoided)"
                )

    # =======================================================================
    # TAB 4: ECONOMICS
    # =======================================================================
    with tab_economics:
        st.subheader("Unit Economics: Savings Waterfall & Conversion Guardrail")
        st.caption(
            "Monthly financial bridge (`q('waterfall')`, `q('savings_kpi')`, `q('tier_detail')`) scaled 3,200x from 25,000 synthetic orders to 80M monthly orders."
        )

        # Top Economics KPI Cards
        ek1, ek2, ek3, ek4 = st.columns(4, gap="medium")
        with ek1:
            with st.container(border=True):
                st.markdown(
                    render_badge("NET MONTHLY BENEFIT", "rto-badge--emerald"),
                    unsafe_allow_html=True,
                )
                st.metric(
                    "Modeled Net Monthly Savings",
                    f"Rs.{kpi['net_savings_cr']:.1f} Cr/mo",
                    delta=f"{kpi['pct_of_baseline']:.1f}% of Rs.{top['loss_cr_month']:.1f} Cr baseline loss",
                    delta_color="normal",
                )
                st.caption(f"Illustrative annualization: Rs.{12 * kpi['net_savings_cr']:.1f} Cr/year (monthly assumptions held constant)")
        with ek2:
            with st.container(border=True):
                st.markdown(
                    render_badge("GROSS LOGISTICS SAVED", "rto-badge--emerald"),
                    unsafe_allow_html=True,
                )
                st.metric(
                    "Direct RTO Loss Avoided",
                    f"Rs.{kpi['loss_avoided_cr']:.1f} Cr/mo",
                    delta=f"-{kpi['rto_cut_pct']:.1f}% direct RTO loss cut",
                    delta_color="normal",
                )
                st.caption("Residual RTO loss drops from `Rs.122.0 Cr` to `Rs.103.3 Cr/mo`")
        with ek3:
            with st.container(border=True):
                st.markdown(
                    render_badge("GUARDRAIL PASS (<1.50%)" if kpi['conversion_drop_pct'] < 1.5 else "GUARDRAIL BREACH", "rto-badge--emerald" if kpi['conversion_drop_pct'] < 1.5 else "rto-badge--red"),
                    unsafe_allow_html=True,
                )
                st.metric(
                    "Blended Conversion Drop",
                    f"{kpi['conversion_drop_pct']:.2f}%",
                    delta=f"{1.50 - kpi['conversion_drop_pct']:.2f}pp headroom below 1.50% max",
                    delta_color="normal",
                )
                st.caption("Take-rate opportunity cost: `-Rs.2.4 Cr/month` (at 10% take-rate)")
        with ek4:
            with st.container(border=True):
                st.markdown(
                    render_badge("MODELED TARGET MET (>=Rs.15 Cr)" if kpi['net_savings_cr'] >= 15 else "MODELED TARGET MISSED", "rto-badge--emerald" if kpi['net_savings_cr'] >= 15 else "rto-badge--red"),
                    unsafe_allow_html=True,
                )
                st.metric(
                    "V3 Net Savings Target (Rs.15 Cr)",
                    f"+Rs.{kpi['net_savings_cr'] - 15.0:.1f} Cr vs KR",
                    delta=f"{100 * kpi['net_savings_cr'] / 15:.1f}% of assumed target (simulation)",
                    delta_color="normal",
                )
                st.caption("Net fee/reward balance: `+Rs.24.3 Cr` fee vs `-Rs.10.8 Cr` UPI disc.")

        # Waterfall Chart & Bridge Table
        w_left, w_right = st.columns([1.3, 1.0], gap="medium")
        with w_left:
            with st.container(border=True):
                st.markdown("#### Monthly Unit-Economics Bridge (`q('waterfall')`)")
                st.caption(
                    "Benefits and costs are signed from the modeled savings perspective. Baseline burden and net total are reference bars; do not add them to the movements. Rounded components may not sum exactly to the canonical net total."
                )
                render_waterfall_chart(wf_df)

        with w_right:
            with st.container(border=True):
                st.markdown("#### Bridge Component Breakdown")
                st.caption(
                    "All figures in `Rs. Crore / month` across `80M` monthly orders (`scale = 3,200x`)."
                )
                wf_disp = wf_df.copy()
                wf_disp["pct_of_base"] = (
                    wf_disp["value_cr"].abs() / float(top["loss_cr_month"]) * 100.0
                ).round(1)
                wf_disp["direction"] = [
                    "Baseline Burden",
                    "Gross Logistics Savings (+)",
                    "Prepaid Incentive Cost (-)",
                    "COD Surcharge Offset (+)",
                    "Lost Take-Rate Margin (-)",
                    "Net Monthly Savings (=)",
                ]
                st.dataframe(
                    wf_disp[["component", "direction", "value_cr", "pct_of_base"]],
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "component": st.column_config.TextColumn("Waterfall Component"),
                        "direction": st.column_config.TextColumn("Economic Role"),
                        "value_cr": st.column_config.NumberColumn(
                            "Monthly Impact", format="Rs. %+.1f Cr"
                        ),
                        "pct_of_base": st.column_config.NumberColumn(
                            "% of Baseline", format="%.1f%%"
                        ),
                    },
                )

        # Per-Tier Economics Table (q('tier_detail'))
        with st.container(border=True):
            st.markdown("#### Per-Tier Intervention Economics & Guardrail Attribution (`q('tier_detail')`)")
            st.caption(
                "Notice that **High Risk (`35–60%` RTO)** generates the largest net unlock (`Rs.24.2 Cr/mo`) via the combined UPI discount nudge and Rs.30 COD fee, while **Medium Risk (`15–35%` RTO)** delivers `Rs.5.4 Cr/mo` in pure logistics savings with zero conversion loss."
            )
            st.dataframe(
                tier_df[
                    [
                        "tier",
                        "orders",
                        "scaled_orders_m",
                        "order_share_pct",
                        "base_rto_pct",
                        "post_rto_pct",
                        "avoided_cr",
                        "reward_cr",
                        "surcharge_cr",
                        "conv_cost_cr",
                        "net_savings_cr",
                        "intervention_policy",
                    ]
                ],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "tier": st.column_config.TextColumn("Risk Tier"),
                    "orders": st.column_config.NumberColumn("Sample (n)", format="%d"),
                    "scaled_orders_m": st.column_config.NumberColumn("Volume (M/mo)", format="%.2fM"),
                    "order_share_pct": st.column_config.NumberColumn("Share (%)", format="%.1f%%"),
                    "base_rto_pct": st.column_config.NumberColumn("Base RTO", format="%.1f%%"),
                    "post_rto_pct": st.column_config.NumberColumn("Post RTO", format="%.1f%%"),
                    "avoided_cr": st.column_config.NumberColumn("RTO Avoided", format="Rs. %.1f Cr"),
                    "reward_cr": st.column_config.NumberColumn("UPI Disc. Cost", format="Rs. %.1f Cr"),
                    "surcharge_cr": st.column_config.NumberColumn("COD Fee Rev.", format="Rs. %.1f Cr"),
                    "conv_cost_cr": st.column_config.NumberColumn("Conv. Cost", format="Rs. %.1f Cr"),
                    "net_savings_cr": st.column_config.NumberColumn("Net Benefit", format="Rs. %.1f Cr"),
                    "intervention_policy": st.column_config.TextColumn("Intervention Rule"),
                },
            )

    # =======================================================================
    # TAB 5: PRIORITIZATION (RICE ROADMAP)
    # =======================================================================
    with tab_roadmap:
        st.subheader("Prioritization: RICE-Ranked Product & Platform Initiatives")
        st.caption(
            "Product roadmap (`q('rice')`) ranked by `RICE = (Reach x Impact x Confidence) / Effort`. "
            "Reach is indexed in `10M monthly orders` touched (`1–10`), Impact on a `0.25–3.0x` scale, Confidence as a `0–100%` probability, and Effort in `engineering person-months`."
        )

        # Roadmap Filter Controls + Reset
        with st.container(border=True):
            rf_col1, rf_col2, rf_col3 = st.columns([2.0, 1.2, 0.8], gap="medium")
            with rf_col1:
                sel_phases = st.multiselect(
                    "Filter by Rollout Horizon (Phase)",
                    options=["V2 (60-90d)", "V3 (6mo)", "V4 (9mo)", "V4 (12mo)", "V5 (18mo)"],
                    key="rice_phases",
                )
            with rf_col2:
                sel_min_conf = st.slider(
                    "Minimum Confidence Threshold",
                    min_value=0.50,
                    max_value=0.95,
                    step=0.05,
                    key="rice_min_conf",
                    help="Filter initiatives by minimum confidence score (0.50 = 50% to 0.90 = 90%).",
                )
            with rf_col3:
                st.write("")
                st.button(
                    "Reset Roadmap Filters",
                    on_click=reset_roadmap_filters,
                    use_container_width=True,
                )

        filtered_rice = filter_rice_roadmap(
            rice_raw_df, phases=sel_phases, min_confidence=float(sel_min_conf)
        )

        if filtered_rice.empty:
            st.warning(
                "**No Roadmap Initiatives Match Current Filters:** Lower the minimum confidence slider or restore rollout phases using **Reset Roadmap Filters**."
            )
        else:
            rk1, rk2, rk3, rk4 = st.columns(4, gap="medium")
            top_init = filtered_rice.iloc[0]
            rk1.metric(
                "Initiatives in Scope",
                f"{len(filtered_rice)} / {len(rice_raw_df)}",
                delta=f"Top priority: {top_init['initiative']}",
                delta_color="off",
            )
            rk2.metric(
                "Highest RICE Score",
                f"{top_init['rice']:.1f}",
                delta=f"Horizon: {top_init['phase']}",
                delta_color="normal",
            )
            rk3.metric(
                "Cumulative Effort in Scope",
                f"{filtered_rice['effort'].sum():.1f} person-mo",
                delta=f"Avg {filtered_rice['effort'].mean():.1f} person-mo / initiative",
                delta_color="off",
            )
            rk4.metric(
                "Mean Initiative Confidence",
                f"{filtered_rice['confidence_pct'].mean():.1f}%",
                delta=f"Min threshold: {int(sel_min_conf*100)}%",
                delta_color="off",
            )

            r_left, r_right = st.columns([1.15, 1.35], gap="medium")
            with r_left:
                with st.container(border=True):
                    st.markdown("#### RICE Score Ranking by Rollout Horizon")
                    render_rice_chart(filtered_rice)

            with r_right:
                with st.container(border=True):
                    st.markdown("#### Prioritized Initiative Scorecard (`q('rice')`)")
                    st.dataframe(
                        filtered_rice[
                            [
                                "rank",
                                "initiative",
                                "phase",
                                "priority_band",
                                "reach",
                                "impact",
                                "confidence_pct",
                                "effort",
                                "rice",
                            ]
                        ],
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "rank": st.column_config.NumberColumn("Rank", format="#%d"),
                            "initiative": st.column_config.TextColumn("Initiative"),
                            "phase": st.column_config.TextColumn("Horizon"),
                            "priority_band": st.column_config.TextColumn("Priority Tier"),
                            "reach": st.column_config.NumberColumn("Reach (10M/mo)", format="%.1f"),
                            "impact": st.column_config.NumberColumn("Impact (0.25–3x)", format="%.1fx"),
                            "confidence_pct": st.column_config.NumberColumn("Confidence", format="%d%%"),
                            "effort": st.column_config.NumberColumn("Effort (p-m)", format="%.1f"),
                            "rice": st.column_config.ProgressColumn(
                                "RICE Score", format="%.1f", min_value=0, max_value=15
                            ),
                        },
                    )

            render_callout(
                "Sequencing Rationale — Compounding the V1 Checkout Risk Signal:",
                "V2 (NDR Intelligence Loop, RICE 13.5) and V3 (Dynamic Logistics Routing, RICE 12.8) consume the real-time checkout risk score without adding buyer friction—routing high-risk COD parcels to faster courier lanes and triggering instant WhatsApp NDR re-attempts before parcels turn into Rs.100 RTO losses. Longer-horizon bets like BNPL (RICE 3.6) and RTO Seller Insurance (RICE 2.0) require 3.0 person-months and external underwriting, placing them in V5 (18mo).",
                variant="rto-callout--emerald",
                badge_html=render_badge("ROADMAP SEQUENCING", "rto-badge--emerald"),
            )


if __name__ == "__main__":
    main()
