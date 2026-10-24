"""Meesho RTO Risk Engine app - diagnostic teardown, live checkout simulator, savings waterfall, RICE roadmap.

Self-bootstrapping: if rto_engine.db / meesho_orders.csv are missing, the deterministic generator runs
first. Every displayed metric is computed at runtime from the project tables via queries.sql.
"""
import os
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
os.chdir(BASE)

import risk_engine as eng


def ensure_data():
    """Bootstraps the data layer: generates CSV + DB on first run instead of raising."""
    if not (BASE / 'rto_engine.db').exists() or not (BASE / 'meesho_orders.csv').exists():
        import generate_data
        generate_data.main()


def named_queries():
    qs = {}
    for chunk in (BASE / 'queries.sql').read_text().split(';'):
        m = re.search(r'--\s*name:\s*(\w+)', chunk)
        if m:
            qs[m.group(1)] = chunk.strip()
    return qs


@st.cache_data(show_spinner=False)
def q(name):
    con = sqlite3.connect(BASE / 'rto_engine.db')
    try:
        return pd.read_sql(named_queries()[name], con)
    finally:
        con.close()


st.set_page_config(page_title='Meesho RTO Risk Engine', layout='wide')
st.title('Meesho RTO Risk Engine - Predict / Prevent / Protect')
st.caption('Risk-tiered checkout intelligence for COD. All metrics computed live from rto_engine.db (25k-order '
           'deterministic sample scaled 3,200x to 80M monthly orders).')
ensure_data()

t1, t2, t3, t4 = st.tabs(['1 - Diagnostic Teardown', '2 - Live Checkout Simulator',
                          '3 - Savings Waterfall', '4 - RICE Roadmap'])

with t1:
    top, foc = q('topline').iloc[0], q('focus_split').set_index('focus')
    k, rest = foc.loc[1], foc.loc[0]
    c = st.columns(4)
    c[0].metric('Blended RTO rate', f'{top.blended_rto:.1%}', f'COD {top.cod_rto:.1%} / prepaid {top.prepaid_rto:.1%}')
    c[1].metric('Direct RTO loss', f'Rs {top.loss_cr_month:.0f} Cr/month')
    c[2].metric('Focus-segment loss share', f'{k.loss_share_pct:.0f}%')
    c[3].metric('Focus RTO rate', f'{k.rto_rate:.0%}', f'vs {rest.rto_rate:.0%} everywhere else')
    st.info(f'**80/20 finding:** first-time x Tier-3+ x Fashion x COD orders are {k.order_share_pct:.0f}% of the '
            f'book but drive **{k.loss_share_pct:.0f}% of RTO loss (Rs {k.loss_cr_month:.0f} Cr/month)** - '
            f'this is a checkout-gating problem, not a logistics one.')
    dim = q('by_dimension')
    for col, dname in zip(st.columns(3), ['payment_mode', 'city_tier', 'category']):
        col.write(f'**Loss share - {dname.replace("_", " ")}**')
        col.bar_chart(dim[dim.dimension == dname].set_index('slice')['loss_share_pct'])
    st.write('**Loss concentration by risk-score decile** (bars = decile share, line-friendly cumulative in table)')
    p = q('pareto').set_index('decile')
    st.bar_chart(p['cum_loss_share_pct'])
    st.dataframe(p, use_container_width=True)

with t2:
    st.write('Score a live checkout session and see exactly which intervention fires and what it is worth.')
    l, r = st.columns((2, 3))
    tier = l.selectbox('City tier', [1, 2, 3, 4], index=2)
    cat = l.selectbox('Category', ['Fashion', 'Home', 'Beauty', 'Electronics'])
    aov = l.slider('Order value (Rs)', 120, 3000, 550, step=10)
    first = l.radio('Buyer vintage', ['First-time', 'Repeat']) == 'First-time'
    cod = l.radio('Payment intent', ['COD', 'Prepaid (UPI)']) == 'COD'
    new_seller = l.checkbox('New / unverified seller (<30 days)', value=True)
    pin = l.slider('PIN-code deliverability score', 0.05, 0.95, 0.35, 0.05)
    prof = eng.score_profile(city_tier=tier, category=cat, aov=aov, first_order=first,
                             cod=cod, seller_new=new_seller, pin=pin)
    r.metric('Risk score (0-100)', prof['risk_score'], f"{prof['risk_prob']:.1%} predicted RTO")
    r.write(f"**Tier:** {prof['tier']}  \n**Checkout action:** {prof['intervention']}")
    cc = r.columns(3)
    cc[0].metric('RTO after intervention', f"{prof['post_prob']:.1%}", f"{(prof['post_prob']-prof['risk_prob']):+.1%}")
    cc[1].metric('Expected loss avoided', f"Rs {prof['loss_avoided']:.0f}")
    cc[2].metric('Incentive / fee', f"Rs {prof['discount_cost']:.0f} discount" if prof['discount_cost']
                 else (f"Rs {prof['surcharge']:.0f} surcharge" if prof['surcharge'] else 'none'))
    if prof['lost_frac']:
        r.warning(f"Guardrail: {prof['lost_frac']:.0%} of sessions in this tier are expected to abandon checkout.")

with t3:
    wf = q('waterfall')
    st.bar_chart(wf.set_index('component')['value_cr'])
    st.dataframe(wf.set_index('component'), use_container_width=True)
    k = q('savings_kpi').iloc[0]
    c = st.columns(4)
    c[0].metric('Net monthly savings', f'Rs {k.net_savings_cr:.0f} Cr', f'{k.pct_of_baseline:.0f}% of baseline loss')
    c[1].metric('RTO loss avoided', f'Rs {k.loss_avoided_cr:.0f} Cr/month', f'-{k.rto_cut_pct:.0f}% RTO loss')
    c[2].metric('Conversion drop', f'{k.conversion_drop_pct:.2f}%',
                'guardrail < 1.5% - PASS' if k.conversion_drop_pct < 1.5 else 'GUARDRAIL BREACH')
    c[3].metric('V3 KR target (Rs 15 Cr/month)', 'MET' if k.net_savings_cr >= 15 else 'MISSED',
                f'{k.net_savings_cr - 15:+.0f} Cr vs target')
    st.dataframe(q('tier_detail').set_index('tier'), use_container_width=True)
    st.caption('Bridge: baseline direct RTO loss -> loss avoided by tiered interventions -> UPI discount cost '
               '-> COD surcharge revenue -> conversion-opportunity cost -> net. Blended across all orders.')

with t4:
    rice = q('rice')
    st.write('**Platform-extension roadmap ranked by RICE = Reach x Impact x Confidence / Effort** '
             '(inputs from case-study Table 9; scores computed live in queries.sql).')
    st.bar_chart(rice.set_index('initiative')['rice'])
    st.dataframe(rice.set_index('initiative'), use_container_width=True)
    st.caption('NDR Intelligence Loop and Dynamic Logistics Routing lead: they reuse the same risk signal the '
               'checkout engine already produces, so they compound the savings waterfall above.')
