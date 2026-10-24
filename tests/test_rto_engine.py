"""Determinism, concentration and guardrail tests for the Meesho RTO Risk Engine."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate_data import MONTHLY_ORDERS, N, build          # noqa: E402
from risk_engine import score_orders, score_profile, simulate  # noqa: E402

SCALE = MONTHLY_ORDERS / N  # 3,200 population orders per sampled order


def focus_mask(df):
    return (df.payment_mode == 'COD') & (df.first_order == 1) & (df.city_tier >= 3) & (df.category == 'Fashion')


def test_generator_is_deterministic():
    assert build().equals(build())


def test_scores_bounded_and_calibrated():
    s = score_orders(build())
    assert s.risk_score.between(0, 100).all()
    assert set(s.risk_tier) <= {'Low', 'Medium', 'High', 'Very High'}
    # calibration: predicted probability tracks realized RTO within 2pp
    assert abs(s.risk_prob.mean() - s.rto.mean()) < 0.02


def test_focus_segment_drives_about_65pct_of_loss():
    s = score_orders(build())
    f = focus_mask(s)
    assert 0.20 < f.mean() < 0.35                          # segment is a large minority of the order book
    share = s.loc[f, 'rto_loss'].sum() / s.rto_loss.sum()
    assert 0.60 < share < 0.72, share                        # case-study 80/20 finding: 60-70%


def test_monthly_loss_drag_scales_to_about_120cr():
    s = score_orders(build())
    cr = s.rto_loss.sum() * SCALE / 1e7
    assert 105 < cr < 135, cr


def test_net_savings_beat_15cr_under_1_5pct_conversion_drop():
    _, tot = simulate(score_orders(build()))
    net_cr = tot['net_savings'] * SCALE / 1e7
    assert net_cr >= 15, net_cr                             # V3 KR: >= Rs 15 Cr/month net savings
    assert tot['conv_drop'] < 0.015, tot['conv_drop']       # guardrail: < 1.5% blended conversion drop
    assert tot['rto_avoided'] * SCALE / 1e7 >= 15           # pure logistics loss avoided also clears the bar


def test_simulator_extremes():
    risky = score_profile(city_tier=3, category='Fashion', aov=600, first_order=True,
                          cod=True, seller_new=True, pin=0.15)
    safe = score_profile(city_tier=1, category='Home', aov=1500, first_order=False,
                         cod=False, seller_new=False, pin=0.90)
    assert risky['risk_score'] > 60 and risky['tier'] in ('High', 'Very High')
    assert safe['risk_score'] < 15 and safe['tier'] == 'Low'
    assert risky['loss_avoided'] > 0
