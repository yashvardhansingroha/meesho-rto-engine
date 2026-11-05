# Meesho RTO Risk Engine

Predict / Prevent / Protect — an independent, synthetic case-study workspace for modeling COD delivery
risk before checkout. Built from the Meesho RTO case study: score every order 0-100, then simulate interventions with
friction proportional to risk (WhatsApp confirmation, Rs.40 UPI discount vs Rs.30 COD surcharge,
prepaid-only for the extreme tail) instead of blanket COD fees that punish honest buyers.

This is not Meesho production data or a deployed checkout intervention. The scorecard weights are
specified assumptions, not coefficients fitted by machine learning. Modeled savings are not observed business impact.

**Live workspace:** https://yashvardhansingroha-meesho-rto-engine-app-ft8aru.streamlit.app/

## Reproducible synthetic scenario

On a deterministic 25,000-order sample (seed 42) scaled 3,200x to 80M monthly platform orders:

| Metric | Value (computed, reproducible) |
|---|---|
| Blended RTO rate (COD / prepaid) | 15.2% (18.3% / 4.1%) |
| Direct RTO loss | **Rs 122 Cr/month** |
| Focus segment: first-time x Tier-3+ x Fashion x COD | 29.3% of orders, **65.8% of loss** |
| Focus-segment RTO vs rest | 34.3% vs 7.4% |
| Modeled engine net savings | **Rs 29.9 Cr/month** (Rs 18.7 Cr modeled loss avoided) |
| Assumed-behavior blended conversion drop | **0.56%** (guardrail < 1.5%) |

## Repo structure

| File | Purpose |
|---|---|
| `generate_data.py` | Deterministic 25k-order generator -> `meesho_orders.csv` + `rto_engine.db` (tables: orders, scores, sim, initiatives, meta) |
| `risk_engine.py` | Logistic scorecard (0-100), risk tiers, intervention simulator, savings math |
| `queries.sql` | Named analytics queries: loss concentration, unit-economics waterfall, RICE. The app runs these same queries |
| `app.py` | Five-section Streamlit workspace with segment filters, checkout presets, score explanations and reset controls. Regenerates data if the DB/CSV are missing |
| `.streamlit/config.toml` | Dark workspace theme |
| `tests/` | Core synthetic-model checks plus real Streamlit interaction tests and display/engine consistency checks |

## Quickstart

```bash
pip install -r requirements.txt
python generate_data.py          # build meesho_orders.csv + rto_engine.db (or let app.py do it)
streamlit run app.py             # 5-section workspace
sqlite3 rto_engine.db < queries.sql   # run the analytics layer standalone
python -m pytest -q              # 14 tests
```

## The five sections

1. **Overview & Headline KPIs** — synthetic-book scale, loss concentration, modeled savings and
   intervention policy, with visible assumptions and denominator definitions.
2. **Risk Diagnosis & Concentration** — filter payment mode, geography, category, buyer vintage,
   seller tenure and risk tier; isolate the focus cohort or reset to the full book.
3. **Checkout Simulator** — four preset profiles and editable buyer/order inputs; inspect scorecard
   log-odds contributions, the proposed intervention and per-order expected economics.
4. **Unit Economics & Waterfall** — modeled benefits/costs, tier-level economics, conversion
   guardrail and Rs 15 Cr/month scenario target. Loss avoided is displayed as a positive benefit;
   baseline burden and net total are reference bars, not additive movements. SQL results stay unchanged.
5. **RICE Prioritization** — eight proposed extensions ranked by Reach x Impact x Confidence / Effort,
   with phase/confidence filters and reset controls.

## How the engine works

- **Score**: logistic scorecard on COD intent, first-time buyer, Tier-3+ PIN, fashion category,
  Rs 300-900 impulse band, new seller, first-time x COD interaction and PIN deliverability.
  `risk_score = 100 x specified-scorecard RTO probability`. Agreement within 2pp on this constructed
  sample is a consistency check, not evidence of out-of-sample predictive calibration.
- **Tier & act** (case-study cutoffs): <0.15 Low = standard COD; 0.15-0.35 Medium = WhatsApp
  confirmation; 0.35-0.60 High = UPI discount lead / COD surcharge fallback; >0.60 Very High =
  prepaid-only.
- **Simulate**: expected-value simulation of each intervention (prepaid takeup removes the COD logit
  terms; surcharge and confirmation cut residual RTO; abandonment is charged take-rate margin on lost
  GMV) -> net savings Rs 29.9 Cr/month at 0.56% blended conversion drop.

## Assumptions (all documented in code)

- Rs 100 direct loss per RTO (Rs 60 forward + Rs 40 reverse), 10% take rate — case-study table 3.
- Intervention behavior constants (takeup 22%/65%, within-tier drops 2.5%/35%, 12%/10% relative RTO
  cuts) are case-study assumptions. References to A/B arms in the source comments are scenario inputs,
  not evidence that this repository ran a real experiment.
- RICE reach/impact/confidence/effort are the case study's relative scores; the score itself is
  computed live in SQL.
- The order book is simulated but **deterministic**: same seed (42), same numbers, every time.
- The generator intentionally embeds the focus-segment correlations. Its concentration result is
  constructed evidence for the case study, not an independently discovered market statistic.
- Scaling a sample to 80M orders assumes the same mix and behavior. No sampling uncertainty,
  operational capacity limits or real-world causal response is estimated here.
- The UI redesign preserves the existing engine, generator, SQL, CSV and database; explanations and
  display-only transformations are checked against those canonical sources.
