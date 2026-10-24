# Meesho RTO Risk Engine

Predict / Prevent / Protect — a risk-tiered checkout engine that prices COD delivery risk **before** the
loss happens. Built from the Meesho RTO case study: score every order 0-100, then intervene with
friction proportional to risk (WhatsApp confirmation, Rs.40 UPI discount vs Rs.30 COD surcharge,
prepaid-only for the extreme tail) instead of blanket COD fees that punish honest buyers.

## The finding this repo proves

On a deterministic 25,000-order sample (seed 42) scaled 3,200x to 80M monthly platform orders:

| Metric | Value (computed, reproducible) |
|---|---|
| Blended RTO rate (COD / prepaid) | 15.2% (18.3% / 4.1%) |
| Direct RTO loss | **Rs 122 Cr/month** |
| Focus segment: first-time x Tier-3+ x Fashion x COD | 29.3% of orders, **65.8% of loss** |
| Focus-segment RTO vs rest | 34.3% vs 7.4% |
| Engine net savings | **Rs 29.9 Cr/month** (Rs 18.7 Cr pure loss avoided) |
| Blended conversion drop | **0.56%** (guardrail < 1.5%) |

## Repo structure

| File | Purpose |
|---|---|
| `generate_data.py` | Deterministic 25k-order generator -> `meesho_orders.csv` + `rto_engine.db` (tables: orders, scores, sim, initiatives, meta) |
| `risk_engine.py` | Logistic scorecard (0-100), risk tiers, intervention simulator, savings math |
| `queries.sql` | Named analytics queries: loss concentration, unit-economics waterfall, RICE. The app runs these same queries |
| `app.py` | Streamlit app, 4 tabs. Self-bootstraps: regenerates data if the DB/CSV are missing |
| `tests/` | Determinism, calibration, concentration and guardrail tests |

## Quickstart

```bash
pip install -r requirements.txt
python generate_data.py          # build meesho_orders.csv + rto_engine.db (or let app.py do it)
streamlit run app.py             # 4-tab app
sqlite3 rto_engine.db < queries.sql   # run the analytics layer standalone
python -m pytest -q              # 6 tests
```

## The four tabs

1. **Diagnostic Teardown** — where the loss actually sits: focus-segment concentration, loss share by
   payment mode / tier / category, cumulative loss by risk-score decile.
2. **Live Checkout Simulator** — pick buyer vintage, tier, category, AOV, seller tenure and PIN
   deliverability; get the risk score, tier, the exact checkout intervention and its expected value.
3. **Savings Waterfall** — Rs Cr/month bridge: baseline loss -> loss avoided -> UPI discount cost ->
   COD surcharge revenue -> conversion cost -> net, with the <1.5% guardrail and Rs 15 Cr V3 target.
4. **RICE Roadmap** — 8 platform extensions ranked by Reach x Impact x Confidence / Effort.

## How the engine works

- **Score**: logistic scorecard on COD intent, first-time buyer, Tier-3+ PIN, fashion category,
  Rs 300-900 impulse band, new seller, first-time x COD interaction and PIN deliverability.
  `risk_score = 100 x predicted RTO probability` (calibrated within 2pp on the sample).
- **Tier & act** (case-study cutoffs): <0.15 Low = standard COD; 0.15-0.35 Medium = WhatsApp
  confirmation; 0.35-0.60 High = UPI discount lead / COD surcharge fallback; >0.60 Very High =
  prepaid-only.
- **Prove**: expected-value simulation of each intervention (prepaid takeup removes the COD logit
  terms; surcharge and confirmation cut residual RTO; abandonment is charged take-rate margin on lost
  GMV) -> net savings Rs 29.9 Cr/month at 0.56% blended conversion drop.

## Assumptions (all documented in code)

- Rs 100 direct loss per RTO (Rs 60 forward + Rs 40 reverse), 10% take rate — case-study table 3.
- Intervention behavior constants (takeup 22%/65%, within-tier drops 2.5%/35%, 12%/10% relative RTO
  cuts) anchored to the case-study A/B arms (control 28.5% RTO, hybrid arm -39.6% RTO, -5.5% conv).
- RICE reach/impact/confidence/effort are the case study's relative scores; the score itself is
  computed live in SQL.
- The order book is simulated but **deterministic**: same seed (42), same numbers, every time.
