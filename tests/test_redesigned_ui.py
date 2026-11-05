"""Exercise the real Streamlit app, reset callbacks and display-only calculations."""
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

import app as ui
from risk_engine import score_profile, sigmoid

ROOT = Path(__file__).resolve().parents[1]


def rendered_app():
    instance = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=120)
    assert not instance.exception
    return instance


def press(instance, label):
    next(button for button in instance.button if button.label == label).click().run(timeout=120)
    assert not instance.exception


@pytest.mark.parametrize('name', list(ui.SIMULATOR_PRESETS))
def test_score_explanation_matches_engine(name):
    preset = ui.SIMULATOR_PRESETS[name]
    args = dict(city_tier=preset['sim_city_tier'], category=preset['sim_category'], aov=preset['sim_aov'],
                first_order=preset['sim_buyer_vintage'] == 'First-time', cod=preset['sim_payment_intent'] == 'COD',
                seller_new=preset['sim_seller_new'], pin=preset['sim_pin'])
    profile = score_profile(**args)
    contributions = ui.compute_scorecard_contributions(**args)
    assert np.isclose(sigmoid(contributions['Log-Odds Impact'].sum()), profile['risk_prob'], atol=1e-6)
    economics = ui.compute_simulator_economics(profile, args['aov'])
    expected = profile['loss_avoided'] + profile['surcharge'] - profile['discount_cost'] - profile['lost_frac'] * 0.1 * args['aov']
    assert economics['net_unit_value_rs'] == round(expected, 2)


def test_filter_summary_matches_original_sql_focus():
    orders = ui.load_enriched_orders()
    chosen = ui.filter_diagnosis_orders(orders, ['COD'], [3, 4], ['Fashion'], ['First-time'],
                                       ui.DIAGNOSIS_FILTER_DEFAULTS['diag_seller'], ui.TIER_ORDER)
    result = ui.compute_filtered_diagnosis_summary(chosen, len(orders), orders.rto_loss.sum(), ui.load_meta()['scale'])
    original = ui.q('focus_split').set_index('focus').loc[1]
    assert result['sample_orders'] == original.orders
    assert result['loss_share_pct'] == original.loss_share_pct
    assert result['order_share_pct'] == original.order_share_pct
    assert ui.filter_diagnosis_orders(orders, [], [3], ['Fashion'], ['First-time'], ['Established Seller'], ui.TIER_ORDER).empty


def test_cost_bridge_sign_does_not_mutate_sql_result():
    original = ui.q('waterfall')
    snapshot = original.copy(deep=True)
    display = ui.prepare_waterfall_display(original)
    assert original.equals(snapshot)
    assert display.set_index('component').loc['RTO loss avoided', 'value_cr'] > 0
    movements = display[~display.component.isin(['Baseline RTO loss', 'Net monthly savings'])].value_cr.sum()
    total = display.set_index('component').loc['Net monthly savings', 'value_cr']
    assert abs(movements - total) <= 0.2  # component-level one-decimal rounding


def test_all_sections_presets_and_resets_render():
    instance = rendered_app()
    assert len(instance.tabs) == 5
    for name, preset in ui.SIMULATOR_PRESETS.items():
        instance.selectbox(key='sim_preset_selector').select(name).run(timeout=120)
        assert not instance.exception
        assert instance.selectbox(key='sim_city_tier').value == preset['sim_city_tier']
        assert instance.slider(key='sim_aov').value == preset['sim_aov']
    press(instance, 'Reset Simulator to Default')
    assert instance.radio(key='sim_payment_intent').value == 'COD'
    instance.multiselect(key='diag_payment').set_value([]).run(timeout=120)
    assert not instance.exception
    assert any('Empty Filter Slice' in message.value for message in instance.warning)
    press(instance, 'Reset Diagnosis Filters')
    assert instance.multiselect(key='diag_payment').value == ['COD', 'Prepaid']
    press(instance, 'Isolate 80/20 Focus Segment')
    assert instance.multiselect(key='diag_payment').value == ['COD']
    assert instance.multiselect(key='diag_tiers').value == [3, 4]
    assert instance.multiselect(key='diag_categories').value == ['Fashion']
    instance.multiselect(key='rice_phases').set_value([]).run(timeout=120)
    instance.selectbox(key='sim_preset_selector').select('Metro Prepaid Baseline (Low Frictionless)').run(timeout=120)
    press(instance, 'Reset All Workspace Controls')
    assert instance.multiselect(key='diag_payment').value == ['COD', 'Prepaid']
    assert instance.multiselect(key='rice_phases').value == ui.ROADMAP_FILTER_DEFAULTS['rice_phases']
    assert instance.selectbox(key='sim_preset_selector').value == 'Focus Segment (High Risk Default)'


def test_roadmap_empty_filter_and_reset():
    instance = rendered_app()
    instance.multiselect(key='rice_phases').set_value([]).run(timeout=120)
    assert not instance.exception
    assert len(instance.multiselect(key='rice_phases').value) == 0
    press(instance, 'Reset Roadmap Filters')
    assert instance.multiselect(key='rice_phases').value == ui.ROADMAP_FILTER_DEFAULTS['rice_phases']
