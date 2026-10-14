"""Tests for the CN0549 kit.

No CN0549 in this environment, so what is testable is: the IEPE bias
classification (a pure function, and the check that catches the failure the data
cannot show), the configuration, and the experiment code that produces the
kit's headline numbers.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "core"))
sys.path.insert(0, os.path.join(_REPO, "edge"))
sys.path.insert(0, os.path.join(_HERE, "..", "edge"))
sys.path.insert(0, os.path.join(_HERE, "..", "tools"))

from pdmedge.calibration import Calibration  # noqa: E402
from pdmedge.config import load_config  # noqa: E402
from pdmedge.source import SourceError  # noqa: E402

from cn0549_source import (  # noqa: E402
    IEPE_BIAS_MAX_V,
    IEPE_BIAS_MIN_V,
    CN0549Source,
    classify_bias,
)

_CONFIG = os.path.join(_HERE, "..", "config", "agent.cn0549.json")
_CAL = os.path.join(_HERE, "..", "config", "calibration.cn0549.json")


# -------------------------------------------------------------- the bias check
def test_bias_in_band_is_usable():
    for v in (8.0, 10.5, 12.0, 15.9):
        st = classify_bias(v)
        assert st.usable, f"{v} V should be a live sensor, got {st.state}"


def test_open_circuit_is_detected_at_the_supply_rail():
    """A cut cable or unmated connector pulls the line to the supply.

    This is the check that catches the failure the data cannot show: the kit's
    own experiment measures a disconnected sensor being diagnosed as mechanical
    looseness at 97 % confidence.
    """
    st = classify_bias(23.0, supply_v=24.0)
    assert st.state == "open"
    assert not st.usable
    assert "open circuit" in st.detail


def test_short_is_detected():
    st = classify_bias(0.2)
    assert st.state == "short"
    assert not st.usable


def test_out_of_band_is_unknown_not_ok():
    """Between a short and the normal window is ambiguous, and ambiguous must
    not read as a pass: a wrong sensor type or a failing current source lands
    here."""
    st = classify_bias(3.0)
    assert st.state == "unknown"
    assert not st.usable
    st2 = classify_bias(float("nan"))
    assert st2.state == "unknown" and not st2.usable


def test_bias_window_is_inside_the_supply():
    """A window that reached the rail could never detect an open circuit."""
    assert IEPE_BIAS_MAX_V < 24.0 * 0.85


# ----------------------------------------------------------------- the source
def test_filter_type_has_no_default():
    """sinc5 and wideband trade transition sharpness against group delay, and
    neither is safe for every job -- so the constructor refuses a bad one and
    the README has to state the choice."""
    with pytest.raises(SourceError, match="filter_type"):
        CN0549Source(filter_type="whatever")
    src = CN0549Source(filter_type="wideband")
    assert "0.433" in src.FILTERS["wideband"]
    assert "group delay" in src.FILTERS["sinc5"]


def test_describe_names_the_filter_and_the_bias():
    """Both silently change what the numbers mean, so both are printed."""
    text = CN0549Source(filter_type="sinc5").describe()
    assert "sinc5" in text
    assert "IEPE bias" in text
    assert "CHECK THE CABLE" in text, "an unread bias must not look like a pass"


def test_source_reports_the_rate_it_will_use():
    src = CN0549Source(fs_hz=32000.0, filter_type="wideband")
    assert src.fs_hz == 32000.0
    assert "read back" in src.describe() or "32000" in src.describe()


# ---------------------------------------------------------------- the config
def test_shipped_config_loads_and_is_consistent():
    cfg = load_config(_CONFIG)
    assert cfg.tier == "node"
    assert cfg.channels, "no channels configured"
    # The block must be long enough for the envelope FFT: decimation cannot
    # create resolution, only record length can.
    assert cfg.block_samples >= cfg.envelope_n_fft
    assert cfg.block_samples >= cfg.n_fft
    # And the duty cycle must be sane.
    assert cfg.block_samples / cfg.fs_hz < cfg.interval_s


def test_shipped_calibration_is_complete_and_current():
    cal = Calibration.load(_CAL)
    assert len(cal) >= 1
    for name in cal.channels:
        c = cal[name]
        assert c.certificate_date, f"{name} has no certificate date"
        assert c.serial, f"{name} has no serial number"
        assert c.certificate_status() in ("current", "due"), (
            f"{name}: certificate is {c.certificate_status()}"
        )
        # A 100 mV/g piezo on a 4.096 V reference clips at ~41 g. If the
        # configured full scale implied something absurd, every amplitude on
        # this channel would be wrong by that factor.
        assert 10.0 < c.full_scale_g < 500.0


def test_calibration_channels_match_the_config():
    """A config naming a channel the calibration does not have fails at
    runtime, on the first acquisition, in the field."""
    cfg = load_config(_CONFIG)
    cal = Calibration.load(_CAL)
    for ch in cfg.channels:
        if ch.units_are_g:
            continue
        key = ch.source_channel or ch.name
        assert key in cal, f"config channel {ch.name!r} maps to {key!r}, not in the calibration"


def test_noise_budget_says_sensor_limited():
    """The kit's central claim about its own converter, as a test.

    If this ever fails it means the configured chain became converter-limited,
    which would mean the 24-bit part is finally doing something -- and the
    README's argument would need rewriting.
    """
    cal = Calibration.load(_CAL)
    cfg = load_config(_CONFIG)
    for name in cal.channels:
        b = cal[name].noise_budget(cfg.fs_hz / 2.0)
        assert b["limited_by"] == "sensor", f"{name}: {b}"
        assert float(b["margin_db"]) > 30.0, f"{name}: only {b['margin_db']:.0f} dB"


# ------------------------------------------------------------- the experiments
def test_quantisation_is_a_converter_grid():
    from experiments import quantise

    x = np.array([0.0, 0.0001, 0.5, -0.5, 100.0, -100.0])
    q = quantise(x, 16, 16.0)
    lsb = 2.0 * 16.0 / (1 << 16)
    assert np.all(np.abs(q) <= 16.0)
    # Everything lands on the grid.
    assert np.allclose(q / lsb, np.round(q / lsb))
    # And it clips rather than wrapping, which would turn an overload into a
    # spectacular artefact instead of a flat top.
    assert q[4] == pytest.approx(16.0 - lsb)
    assert q[5] == pytest.approx(-16.0)


def test_added_sensor_noise_matches_the_stated_density():
    from experiments import add_sensor_noise

    rng = np.random.default_rng(3)
    fs = 32000.0
    density = 75e-6
    x = add_sensor_noise(np.zeros(200000), density, fs, rng)
    expected_rms = density * np.sqrt(fs / 2.0)
    assert np.std(x) == pytest.approx(expected_rms, rel=0.02)


def test_results_json_is_present_and_self_consistent():
    """The deck's figures are built from this file, so it is checked in."""
    path = os.path.join(_HERE, "..", "results", "experiments.json")
    if not os.path.isfile(path):
        pytest.skip("experiments.json not generated on this checkout")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    r = data["resolution"]
    assert len(r["severities"]) == len(r["median"][next(iter(r["median"]))])
    for name, chain in r["chains"].items():
        # The claim the whole kit argument rests on.
        assert chain["noise_density_g"] > chain["adc_noise_density_g"] * 10.0, (
            f"{name} is not clearly sensor-limited: {chain}"
        )

    d = data["disconnected"]
    assert d["healthy_alarms"] == 0, "the threshold should not alarm on healthy data"
    assert d["faulty_alarms"] == d["n_eval"], "a 0.35 outer-race fault must alarm"
    # The finding: a dead channel alarms too, so the score cannot tell them
    # apart -- which is why the bias check exists.
    assert d["disconnected_alarms"] == d["n_eval"]
    assert d["diagnosis_disconnected"], "no diagnosis recorded for the dead channel"
    top = d["diagnosis_disconnected"][0]
    assert top["confidence"] > 0.5, (
        f"the rule base's reading of converter noise is only {top['confidence']}; "
        f"the README claims a confident wrong diagnosis"
    )
