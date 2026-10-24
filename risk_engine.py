"""Meesho RTO Risk Engine: transparent scorecard risk scoring (0-100) + risk-tiered checkout interventions.

Anchored to the case study (Meesho_RTO_Risk_Engine_Merged): tier cutoffs 0.15/0.35/0.60 on predicted
RTO probability; Medium = WhatsApp confirmation, High = Rs.40 UPI-discount lead with Rs.30 COD-surcharge
fallback, Very High = prepaid-only. All money in INR. Vectorized pandas/numpy only.
"""
import numpy as np
import pandas as pd

# --- Assumptions sourced from the case study (tables 3, 8, 10) ---
RTO_COST, TAKE_RATE = 100.0, 0.10                 # Rs.100 direct loss per RTO (60 fwd + 40 rev); take-rate on GMV
UPI_DISCOUNT, COD_SURCHARGE = 40.0, 30.0          # hybrid nudge: prepaid reward leads, COD surcharge is fallback
CONFIRM_RTO_CUT, SURCHARGE_RTO_CUT = 0.12, 0.10   # relative RTO cut from WhatsApp confirmation / COD surcharge
TAKEUP = {'High': 0.22, 'Very High': 0.65}        # share accepting the UPI discount (A/B arm C prepaid shift 38.5%)
DROP = {'High': 0.025, 'Very High': 0.35}         # within-tier checkout conversion drop (arm C: -5.5% relative)
# Logistic scorecard: risk rises with COD, first-time buyers, Tier-3+ PINs, fashion, the Rs.300-900
# impulse band, new sellers, and the first-time x COD commitment gap; pin_risk = 0.60 - PIN deliverability.
W = dict(base=-4.90, cod=0.70, first=0.75, tier3=0.25, first_tier3=0.60, fashion=0.30, band=0.40,
         seller_new=0.50, cod_first=0.55, pin=2.85)
TIER_CUTS = (0.15, 0.35, 0.60)
INTERVENTION = {'Low': 'Standard COD - zero added friction',
                'Medium': 'Post-order WhatsApp one-tap address/slot confirmation',
                'High': 'Pay via UPI now -> Rs.40 instant discount (else Rs.30 COD surcharge)',
                'Very High': 'Prepaid-only checkout - COD unavailable (Rs.40 UPI discount offered)'}

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def logit(cod, first, tier3, fashion, band, seller_new, pin_risk):
    """Scorecard logit; works on pandas Series or scalars."""
    return (W['base'] + W['cod'] * cod + W['first'] * first + W['tier3'] * tier3
            + W['first_tier3'] * first * tier3 + W['fashion'] * fashion + W['band'] * band
            + W['seller_new'] * seller_new + W['cod_first'] * cod * first + W['pin'] * pin_risk)

def tier_of(p):
    return np.select([p < TIER_CUTS[0], p < TIER_CUTS[1], p < TIER_CUTS[2]],
                     ['Low', 'Medium', 'High'], 'Very High')

def score_orders(df):
    """Attach risk_prob, risk_score (0-100), risk_tier and intervention to an order dataframe."""
    z = logit(df.payment_mode.eq('COD').astype(float), df.first_order.astype(float),
              df.city_tier.ge(3).astype(float), df.category.eq('Fashion').astype(float),
              df.order_value.between(300, 900).astype(float), df.seller_new.astype(float),
              0.60 - df.pin_deliverability)
    out = df.copy()
    out['risk_prob'] = sigmoid(z)
    out['risk_score'] = (100 * out['risk_prob']).round().astype(int)
    out['risk_tier'] = tier_of(out['risk_prob'].to_numpy())
    out['intervention'] = pd.Series(out['risk_tier']).map(INTERVENTION).values
    return out

def prepaid_prob(p, first):
    """Same buyer paying upfront: remove the COD logit effects (commitment + first-time interaction)."""
    return sigmoid(np.log(p / (1 - p)) - W['cod'] - W['cod_first'] * first)

def intervene(s):
    """Expected post-intervention RTO prob, UPI-discount cost, surcharge revenue and lost fraction per order."""
    p, first = s['risk_prob'].to_numpy(), s['first_order'].to_numpy(dtype=float)
    tier = s['risk_tier'].to_numpy() if isinstance(s['risk_tier'], pd.Series) else np.asarray(s['risk_tier'])
    p_post, reward, fee, lost = p.copy(), np.zeros(len(s)), np.zeros(len(s)), np.zeros(len(s))
    m = tier == 'Medium'
    p_post[m] = p[m] * (1 - CONFIRM_RTO_CUT)
    for t in ('High', 'Very High'):
        m = tier == t
        stay = 1.0 - TAKEUP[t] - DROP[t]           # COD stayers (0 for Very High: prepay or abandon)
        p_post[m] = TAKEUP[t] * prepaid_prob(p[m], first[m]) + stay * p[m] * (1 - SURCHARGE_RTO_CUT)
        reward[m] = TAKEUP[t] * UPI_DISCOUNT
        fee[m] = stay * COD_SURCHARGE
        lost[m] = DROP[t]
    return p_post, reward, fee, lost

def simulate(scored):
    """Per-tier expected outcomes plus totals (sample scale; multiply by meta.scale for monthly)."""
    s = scored.copy()
    p_post, reward, fee, lost = intervene(s)
    s = s.assign(p_post=p_post, reward=reward, surcharge_rev=fee, lost=lost,
                 post_loss=p_post * RTO_COST, conv_cost=lost * TAKE_RATE * s.order_value)
    tiers = s.groupby('risk_tier').agg(
        orders=('order_id', 'size'), base_rto=('rto', 'mean'), post_rto=('p_post', 'mean'),
        base_loss=('rto_loss', 'sum'), post_loss=('post_loss', 'sum'), reward=('reward', 'sum'),
        surcharge_rev=('surcharge_rev', 'sum'), orders_lost=('lost', 'sum'),
        conv_cost=('conv_cost', 'sum')).reset_index()
    tot = dict(baseline_loss=s.rto_loss.sum(), rto_avoided=s.rto_loss.sum() - s.post_loss.sum(),
               reward_cost=s.reward.sum(), surcharge_rev=s.surcharge_rev.sum(), conv_cost=s.conv_cost.sum(),
               orders_lost=s.lost.sum(), orders=len(s), base_rto=s.rto.mean(), post_rto=s.p_post.mean(),
               conv_drop=s.lost.mean())
    tot['net_savings'] = tot['rto_avoided'] + tot['surcharge_rev'] - tot['reward_cost'] - tot['conv_cost']
    return tiers, tot

def score_profile(city_tier, category, aov, first_order, cod, seller_new, pin):
    """UI helper: score one hypothetical checkout session end-to-end."""
    z = logit(float(cod), float(first_order), float(city_tier >= 3), float(category == 'Fashion'),
              float(300 <= aov <= 900), float(seller_new), 0.60 - float(pin))
    p = float(sigmoid(z))
    t = str(tier_of(np.array([p]))[0])
    row = pd.DataFrame([dict(risk_prob=p, risk_tier=t, first_order=float(first_order))])
    p_post, reward, fee, lost = intervene(row)
    return dict(risk_score=int(round(100 * p)), risk_prob=p, tier=t, intervention=INTERVENTION[t],
                post_prob=float(p_post[0]), discount_cost=float(reward[0]), surcharge=float(fee[0]),
                lost_frac=float(lost[0]), loss_avoided=(p - float(p_post[0])) * RTO_COST)
