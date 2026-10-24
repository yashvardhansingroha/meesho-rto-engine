"""Deterministic generator: 25,000 sampled orders (one month of platform traffic) -> meesho_orders.csv + rto_engine.db.

Correlations are built in (first-time buyers cluster in Tier-3+ PINs and buy fashion on COD, new sellers
ride along), which is what produces the case-study concentration finding. Seed fixed => every number
reproducible. The population multiplier (80M monthly orders / 25K sample) is stored in meta and used to
scale every Rupee figure to monthly platform level.
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from risk_engine import logit, sigmoid, score_orders, simulate

BASE = Path(__file__).resolve().parent
N, SEED, MONTHLY_ORDERS = 25_000, 42, 80_000_000
OTHER_CATS = ['Home', 'Beauty', 'Electronics']
# RICE inputs from case-study Table 9 (reach/impact/confidence/effort are relative scores; RICE computed in SQL).
INITIATIVES = [('NDR Intelligence Loop', 'V2 (60-90d)', 9.0, 2.5, 0.90, 1.5),
               ('Dynamic Logistics Routing', 'V3 (6mo)', 8.0, 2.0, 0.80, 1.0),
               ('Seller Onboarding Risk Gate', 'V4 (9mo)', 6.0, 3.0, 0.80, 1.5),
               ('Catalog Quality Engine', 'V4 (9mo)', 7.0, 2.0, 0.75, 1.5),
               ('Reseller Network Intelligence', 'V4 (12mo)', 7.0, 2.0, 0.70, 1.5),
               ('BNPL (Meesho Pay Later)', 'V5 (18mo)', 6.0, 3.0, 0.60, 3.0),
               ('Regional Demand Forecasting', 'V5 (18mo)', 5.0, 1.5, 0.80, 2.0),
               ('RTO Seller Insurance', 'V5 (18mo)', 4.0, 3.0, 0.50, 3.0)]


def build():
    rng = np.random.default_rng(SEED)
    tier = rng.choice([1, 2, 3, 4], size=N, p=[0.12, 0.20, 0.45, 0.23])
    t3 = tier >= 3
    first = rng.random(N) < np.where(t3, 0.58, 0.36)
    fashion = rng.random(N) < np.where(t3, 0.80, 0.44)
    cod = rng.random(N) < np.where(t3 & first, 0.93, np.where(first, 0.82, 0.66))
    aov = np.where(fashion, rng.lognormal(6.25, 0.32, N), rng.lognormal(6.10, 0.50, N)).clip(120, 3000).round(-1)
    seller_new = rng.random(N) < (0.22 + 0.18 * (t3 & fashion) + 0.10 * first)
    pin = np.clip(rng.normal(0.62 - 0.17 * t3, 0.17, N), 0.05, 0.95).round(3)
    z = logit(cod.astype(float), first.astype(float), t3.astype(float), fashion.astype(float),
              ((aov >= 300) & (aov <= 900)).astype(float), seller_new.astype(float), 0.60 - pin)
    rto = (rng.random(N) < sigmoid(z + rng.normal(0, 0.38, N))).astype(int)
    cid = np.empty(N, dtype=object)
    cid[first] = ['FT%d' % i for i in np.flatnonzero(first)]
    cid[~first] = ['RC%d' % x for x in rng.choice(np.arange(90000, 96000), size=(~first).sum())]
    return pd.DataFrame(dict(
        order_id=np.array(['MEO-%06d' % i for i in range(N)]),
        order_date=(pd.Timestamp('2026-09-01') + pd.to_timedelta(rng.integers(0, 30, N), unit='D')).astype(str),
        customer_id=cid, first_order=first.astype(int), city_tier=tier,
        category=np.where(fashion, 'Fashion', rng.choice(OTHER_CATS, size=N)),
        payment_mode=np.where(cod, 'COD', 'Prepaid'), order_value=aov, pin_deliverability=pin,
        seller_new=seller_new.astype(int), rto=rto, rto_loss=rto * 100))


def main():
    df, scale = build(), MONTHLY_ORDERS / N
    scored = score_orders(df)
    tiers, tot = simulate(scored)
    df.to_csv(BASE / 'meesho_orders.csv', index=False)
    con = sqlite3.connect(BASE / 'rto_engine.db')
    df.to_sql('orders', con, if_exists='replace', index=False)
    scored[['order_id', 'risk_prob', 'risk_score', 'risk_tier', 'intervention']].to_sql('scores', con, if_exists='replace', index=False)
    tiers.to_sql('sim', con, if_exists='replace', index=False)
    pd.DataFrame(INITIATIVES, columns=['initiative', 'phase', 'reach', 'impact', 'confidence', 'effort']).to_sql('initiatives', con, if_exists='replace', index=False)
    pd.DataFrame(dict(key=['scale', 'monthly_orders', 'n_samples', 'rto_cost', 'take_rate'],
                      value=[scale, MONTHLY_ORDERS, N, 100, 0.10])).to_sql('meta', con, if_exists='replace', index=False)
    con.commit()
    con.close()
    focus = df.payment_mode.eq('COD') & df.first_order.eq(1) & df.city_tier.ge(3) & df.category.eq('Fashion')
    codr = df.loc[df.payment_mode.eq('COD'), 'rto'].mean()
    print(f'orders={len(df)} blended_rto={df.rto.mean():.3f} cod_rto={codr:.3f} '
          f'prepaid_rto={df.loc[df.payment_mode.eq("Prepaid"), "rto"].mean():.3f}')
    print(f'focus_segment: {focus.mean():.1%} of orders, {df.loc[focus, "rto_loss"].sum() / df.rto_loss.sum():.1%} of loss, '
          f'rto={df.loc[focus, "rto"].mean():.3f}')
    print(f'tiers: {scored.risk_tier.value_counts(normalize=True).round(3).to_dict()}')
    print(f'monthly_direct_loss=Rs.{df.rto_loss.sum() * scale / 1e7:.1f}Cr net_savings=Rs.{tot["net_savings"] * scale / 1e7:.1f}Cr '
          f'avoided=Rs.{tot["rto_avoided"] * scale / 1e7:.1f}Cr conv_drop={tot["conv_drop"]:.2%}')


if __name__ == '__main__':
    main()
