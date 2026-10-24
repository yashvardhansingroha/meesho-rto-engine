-- Meesho RTO Risk Engine - analytics layer (SQLite). Run standalone: sqlite3 rto_engine.db < queries.sql
-- Every Rupee figure scales the 25k-order sample to monthly platform volume via meta.scale (80M/25k = 3200).
-- App tabs execute these same named queries (-- name: <id>), so SQL is the single source of truth.

-- name: topline
-- Monthly-scale topline: RTO rates by payment mode and direct loss in Rs Cr/month.
WITH m AS (SELECT CAST(value AS REAL) AS scale FROM meta WHERE key = 'scale')
SELECT COUNT(*) AS orders, ROUND(AVG(rto), 4) AS blended_rto,
       ROUND(AVG(CASE WHEN payment_mode = 'COD' THEN rto END), 4) AS cod_rto,
       ROUND(AVG(CASE WHEN payment_mode = 'Prepaid' THEN rto END), 4) AS prepaid_rto,
       ROUND(SUM(rto_loss) * scale / 1e7, 1) AS loss_cr_month
FROM orders, m;

-- name: focus_split
-- 80/20 concentration: focus segment = first-time x Tier-3+ x Fashion x COD.
WITH m AS (SELECT CAST(value AS REAL) AS scale FROM meta WHERE key = 'scale'),
f AS (SELECT *, (first_order = 1 AND city_tier >= 3 AND category = 'Fashion'
                AND payment_mode = 'COD') AS focus FROM orders, m)
SELECT focus, COUNT(*) AS orders,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS order_share_pct,
       ROUND(AVG(rto), 4) AS rto_rate, ROUND(SUM(rto_loss) * scale / 1e7, 1) AS loss_cr_month,
       ROUND(100.0 * SUM(rto_loss) / SUM(SUM(rto_loss)) OVER (), 1) AS loss_share_pct
FROM f GROUP BY focus;

-- name: by_dimension
-- RTO rate and loss share by payment mode, city tier and category.
SELECT 'payment_mode' AS dimension, payment_mode AS slice, COUNT(*) AS orders,
       ROUND(AVG(rto), 4) AS rto_rate,
       ROUND(100.0 * SUM(rto_loss) / (SELECT SUM(rto_loss) FROM orders), 1) AS loss_share_pct
FROM orders GROUP BY payment_mode
UNION ALL
SELECT 'city_tier', 'Tier ' || city_tier, COUNT(*), ROUND(AVG(rto), 4),
       ROUND(100.0 * SUM(rto_loss) / (SELECT SUM(rto_loss) FROM orders), 1)
FROM orders GROUP BY city_tier
UNION ALL
SELECT 'category', category, COUNT(*), ROUND(AVG(rto), 4),
       ROUND(100.0 * SUM(rto_loss) / (SELECT SUM(rto_loss) FROM orders), 1)
FROM orders GROUP BY category;

-- name: pareto
-- Cumulative RTO loss share by risk-score decile (10 = riskiest orders).
WITH d AS (SELECT NTILE(10) OVER (ORDER BY risk_score) AS decile, o.rto, o.rto_loss
           FROM scores s JOIN orders o USING (order_id))
SELECT decile, COUNT(*) AS orders, ROUND(AVG(rto), 4) AS rto_rate,
       ROUND(100.0 * SUM(rto_loss) / SUM(SUM(rto_loss)) OVER (), 1) AS loss_share_pct,
       ROUND(100.0 * SUM(SUM(rto_loss)) OVER (ORDER BY decile) / SUM(SUM(rto_loss)) OVER (), 1) AS cum_loss_share_pct
FROM d GROUP BY decile ORDER BY decile;

-- name: waterfall
-- Monthly unit-economics bridge of the risk-tiered checkout engine (Rs Cr/month, signed for charting).
WITH m AS (SELECT CAST(value AS REAL) AS scale FROM meta WHERE key = 'scale')
SELECT 'Baseline RTO loss' AS component, ROUND(SUM(base_loss) * scale / 1e7, 1) AS value_cr FROM sim, m
UNION ALL SELECT 'RTO loss avoided', ROUND(-(SUM(base_loss) - SUM(post_loss)) * scale / 1e7, 1) FROM sim, m
UNION ALL SELECT 'UPI discount cost', ROUND(-SUM(reward) * scale / 1e7, 1) FROM sim, m
UNION ALL SELECT 'COD surcharge revenue', ROUND(SUM(surcharge_rev) * scale / 1e7, 1) FROM sim, m
UNION ALL SELECT 'Conversion cost', ROUND(-SUM(conv_cost) * scale / 1e7, 1) FROM sim, m
UNION ALL SELECT 'Net monthly savings',
    ROUND((SUM(base_loss) - SUM(post_loss) + SUM(surcharge_rev) - SUM(reward) - SUM(conv_cost)) * scale / 1e7, 1)
FROM sim, m;

-- name: savings_kpi
-- Headline savings metrics incl. the <1.5% conversion-drop guardrail and the Rs.15 Cr V3 target.
WITH m AS (SELECT CAST(value AS REAL) AS scale FROM meta WHERE key = 'scale'),
t AS (SELECT SUM(base_loss) AS bl, SUM(base_loss) - SUM(post_loss) AS av, SUM(reward) AS rw,
             SUM(surcharge_rev) AS sr, SUM(conv_cost) AS cc,
             SUM(orders_lost) AS lost, SUM(orders) AS n FROM sim)
SELECT ROUND((av + sr - rw - cc) * scale / 1e7, 1) AS net_savings_cr,
       ROUND(av * scale / 1e7, 1) AS loss_avoided_cr,
       ROUND(100.0 * (av + sr - rw - cc) / bl, 1) AS pct_of_baseline,
       ROUND(100.0 * av / bl, 1) AS rto_cut_pct,
       ROUND(100.0 * lost / n, 2) AS conversion_drop_pct
FROM t, m;

-- name: tier_detail
-- Per-tier intervention economics (Rs Cr/month).
WITH m AS (SELECT CAST(value AS REAL) AS scale FROM meta WHERE key = 'scale')
SELECT risk_tier AS tier, orders, ROUND(base_rto, 4) AS base_rto, ROUND(post_rto, 4) AS post_rto,
       ROUND((base_loss - post_loss) * scale / 1e7, 1) AS avoided_cr,
       ROUND(reward * scale / 1e7, 1) AS reward_cr, ROUND(surcharge_rev * scale / 1e7, 1) AS surcharge_cr,
       ROUND(conv_cost * scale / 1e7, 1) AS conv_cost_cr
FROM sim, m
ORDER BY CASE risk_tier WHEN 'Low' THEN 1 WHEN 'Medium' THEN 2 WHEN 'High' THEN 3 ELSE 4 END;

-- name: rice
-- Roadmap ranking: RICE = Reach x Impact x Confidence / Effort (inputs from case-study Table 9).
SELECT initiative, phase, reach, impact, confidence, effort,
       ROUND(reach * impact * confidence / effort, 1) AS rice
FROM initiatives ORDER BY rice DESC;
