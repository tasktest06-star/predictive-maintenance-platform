"""Property tests for whichever hardware kit this branch carries.

Discovers `hardware/*/firmware/app_config.h` rather than naming a kit, so the
same file is valid on every `hw/*` branch and a new kit inherits the checks for
free.

These are not unit tests of the kit's driver — that needs hardware. They are
the checks that catch a *configuration* that will produce plausible, wrong
numbers: an envelope FFT longer than the record it is fed, a demodulation band
above Nyquist, a velocity band that excludes the 1x line, an energy budget whose
dominant term nobody noticed.

Every one of these corresponds to a mistake that was actually made while
building this repository.
"""

from __future__ import annotations

import glob
import os
import re
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "core"))
sys.path.insert(0, os.path.join(_REPO, "tools"))

from pdmcore import FEATURE_SPEC_VERSION  # noqa: E402
from pdmcore.dsp.bearing import BEARING_CATALOG, fault_frequencies  # noqa: E402


#: Accelerometer operating current per kit, mA. The figure that decides whether
#: keeping the sensor powered between acquisitions is ever worth it.
ACCEL_STANDING_MA = {
    "stwinbx1": 1.1,    # IIS3DWB
    "esp32s3": 0.2,     # ADXL355
    "nrf9160": 1.1,     # IIS3DWB
    "alif_e7": 1.1,     # IIS3DWB
    "ra8d1": 1.1,       # IIS3DWB
}


def _configs():
    pattern = os.path.join(_REPO, "hardware", "*", "firmware", "app_config.h")
    return sorted(glob.glob(pattern))


def _parse(path: str) -> dict:
    out = {}
    rx = re.compile(r"^\s*#define\s+(APP_\w+)\s+(.+?)\s*(?:/\*.*)?$")
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = rx.match(line)
            if m:
                out[m.group(1)] = m.group(2).strip()
    return out


def _num(raw: str) -> float:
    raw = raw.strip().rstrip("fu")
    # A few values are expressions like (APP_ENVELOPE_N_FFT * APP_ENVELOPE_DECIMATE).
    if raw.startswith("("):
        return float("nan")
    try:
        return float(int(raw, 0))
    except ValueError:
        return float(raw)


CONFIGS = _configs()
pytestmark = pytest.mark.skipif(not CONFIGS, reason="no hardware/*/firmware/app_config.h on this branch")
IDS = [p.split(os.sep)[-3] for p in CONFIGS]


@pytest.fixture(params=CONFIGS, ids=IDS)
def cfg(request):
    path = request.param
    d = _parse(path)
    d["__kit__"] = path.split(os.sep)[-3]
    d["__path__"] = path
    return d


# ---------------------------------------------------------------------------
def test_feature_spec_matches_pdmcore(cfg):
    """A version mismatch takes the fleet dark: ingest rejects frames it cannot
    lay out rather than mis-scoring them."""
    assert int(_num(cfg["APP_FEATURE_SPEC"])) == FEATURE_SPEC_VERSION


def test_fft_size_is_a_power_of_two(cfg):
    n = int(_num(cfg["APP_N_FFT"]))
    assert n & (n - 1) == 0, f"APP_N_FFT {n} is not a power of two"
    assert 256 <= n <= 8192


def test_acquisition_is_long_enough_for_the_envelope_fft(cfg):
    """Decimation cannot create resolution, only record length can.

    The mistake this catches: a 2048-point envelope FFT configured from a
    4096-sample block (512 samples in). The pipeline refuses and silently stays
    at tier 1, so the bearing block reads as all-zero — indistinguishable from
    "no bearing fault".
    """
    acq = int(_num(cfg["APP_ACQ_SAMPLES"]))
    env_n = int(_num(cfg["APP_ENVELOPE_N_FFT"]))
    dec = int(_num(cfg["APP_ENVELOPE_DECIMATE"]))
    assert env_n * dec <= acq, (
        f"{cfg['__kit__']}: a {env_n}-point envelope FFT decimated by {dec} needs "
        f"{env_n * dec} samples but APP_ACQ_SAMPLES is {acq}"
    )
    assert acq >= int(_num(cfg["APP_N_FFT"]))


def test_demodulation_band_is_below_nyquist(cfg):
    fs = _num(cfg["APP_FS_HZ"])
    hi = _num(cfg["APP_DEMOD_HI_HZ"])
    lo = _num(cfg["APP_DEMOD_LO_HZ"])
    assert 0 < lo < hi < fs / 2, f"{cfg['__kit__']}: band {lo}-{hi} Hz vs Nyquist {fs / 2}"
    # And wide enough that the sidebands of interest are not clipped: a band
    # narrower than a few times the highest modulation rate makes the envelope
    # amplitude speed-dependent.
    assert hi - lo >= 500.0, f"{cfg['__kit__']}: demodulation band is only {hi - lo} Hz wide"


def test_envelope_lowpass_is_below_the_decimated_nyquist(cfg):
    """Otherwise the envelope aliases, and the alias lands on a plausible
    bearing frequency."""
    fs = _num(cfg["APP_FS_HZ"])
    dec = _num(cfg["APP_ENVELOPE_DECIMATE"])
    lp = _num(cfg["APP_ENVELOPE_LP_HZ"])
    assert lp < fs / dec / 2.0, (
        f"{cfg['__kit__']}: envelope LP {lp} Hz vs decimated Nyquist {fs / dec / 2:.0f} Hz"
    )


def test_bearing_harmonics_fit_below_the_decimated_nyquist(cfg):
    fs = _num(cfg["APP_FS_HZ"])
    dec = _num(cfg["APP_ENVELOPE_DECIMATE"])
    harmonics = int(_num(cfg["APP_BEARING_HARMONICS"]))
    geom = BEARING_CATALOG["SKF6205"]
    # Worst case of the configured geometry at a fast machine: BPFI is the
    # highest of the four kinematic frequencies.
    ff = fault_frequencies(geom, 60.0)
    top = ff.bpfi_hz * harmonics
    assert top < fs / dec / 2.0, (
        f"{cfg['__kit__']}: {harmonics} BPFI harmonics reach {top:.0f} Hz at 3600 rpm, "
        f"above the decimated Nyquist {fs / dec / 2:.0f} Hz"
    )


def test_envelope_resolution_resolves_the_fault_frequencies(cfg):
    """The 1.5-bin tolerance floor must stay well inside the ±2 % slip term.

    At 6.5 Hz envelope resolution the floor is ±9 % at BPFO — wide enough to
    match a neighbouring family and report an inner-race fault as an outer-race
    one.
    """
    fs = _num(cfg["APP_FS_HZ"])
    dec = _num(cfg["APP_ENVELOPE_DECIMATE"])
    env_n = _num(cfg["APP_ENVELOPE_N_FFT"])
    df = fs / dec / env_n
    ff = fault_frequencies(BEARING_CATALOG["SKF6205"], 29.6667)
    floor = 1.5 * df / ff.bpfo_hz
    assert floor <= 0.06, (
        f"{cfg['__kit__']}: envelope resolution {df:.2f} Hz gives a ±{floor * 100:.1f} % "
        f"tolerance floor at BPFO {ff.bpfo_hz:.1f} Hz"
    )


def test_velocity_band_includes_the_shaft_line(cfg):
    """ISO 20816's 10–1000 Hz band is correct above 600 rpm. Below that the 1x
    line falls in the stop band and the reading is meaningless — the simulated
    395 rpm agitator reads 1.24 mm/s over 2–1000 Hz and 0.30 over 10–1000."""
    lo = _num(cfg["APP_VELOCITY_LO_HZ"])
    hi = _num(cfg["APP_VELOCITY_HI_HZ"])
    assert 1.0 <= lo <= 10.0
    assert 500.0 <= hi <= 2000.0
    assert hi < _num(cfg["APP_FS_HZ"]) / 2.0


def test_clip_margin_leaves_headroom(cfg):
    """A frame flagged CLIPPED is refused by the cloud, so the data is lost.
    The margin must flag before the ADC actually saturates, not after."""
    margin = _num(cfg["APP_CLIP_MARGIN"])
    assert 0.90 <= margin <= 0.99


def test_battery_thresholds_are_ordered(cfg):
    low = _num(cfg["APP_BATT_LOW_MV"])
    crit = _num(cfg["APP_BATT_CRITICAL_MV"])
    assert crit < low, f"{cfg['__kit__']}: critical {crit} mV is not below low {low} mV"
    # And both above a lithium cell's knee, so the node stops while it can still
    # transmit the fact that it is stopping.
    assert crit >= 3000.0


def test_queue_depth_covers_a_realistic_outage(cfg):
    """A node whose gateway is down for a fortnight must not lose the fortnight
    — whatever took the gateway down may have taken the machine with it."""
    depth = _num(cfg["APP_QUEUE_DEPTH"])
    interval_s = _num(cfg["APP_DEFAULT_INTERVAL_S"])
    days = depth * interval_s / 86400.0
    assert days >= 7.0, (
        f"{cfg['__kit__']}: a {depth:.0f}-frame queue at {interval_s:.0f} s covers only "
        f"{days:.1f} days"
    )


def test_sensor_keepalive_crossover_is_consistent_with_the_energy_model(cfg):
    """`APP_SENSOR_KEEPALIVE_THRESHOLD_S` must be near the interval at which
    leaving the sensor powered becomes cheaper than paying the settling cost.

    Checked against the energy model rather than taken on faith, because this
    constant is exactly the kind of number that gets written once from a rough
    estimate and never revisited.
    """
    import power_budget

    kit_key = {
        "stwinbx1": "stwinbx1", "esp32s3": "esp32s3", "nrf9160": "nrf9160",
        "alif_e7": "alif_e7", "ra8d1": "ra8d1",
    }.get(cfg["__kit__"])
    if kit_key is None or kit_key not in power_budget.KITS:
        pytest.skip(f"{cfg['__kit__']} has no entry in the energy model")
    kit = power_budget.KITS[kit_key]

    # Settling costs settle_ma x settle_ms once per cycle. Keeping the sensor
    # alive costs its standing current continuously. Crossover is where those
    # are equal.
    settle_mas = kit.settle_ma * kit.settle_ms / 1000.0
    standing_ma = ACCEL_STANDING_MA.get(cfg["__kit__"], 1.1)
    crossover_s = settle_mas / standing_ma
    declared = _num(cfg["APP_SENSOR_KEEPALIVE_THRESHOLD_S"])
    assert declared <= max(3.0 * crossover_s, 2.0), (
        f"{cfg['__kit__']}: APP_SENSOR_KEEPALIVE_THRESHOLD_S is {declared:.0f} s but the "
        f"energy model puts the crossover at {crossover_s:.2f} s. Keeping the sensor "
        f"powered above that costs {standing_ma:.2f} mA continuously against "
        f"{settle_mas:.2f} mA.s of settling per cycle -- at the declared threshold that "
        f"is a {declared * standing_ma / settle_mas:.0f}x increase in per-cycle energy."
    )


def test_energy_model_knows_about_this_kit(cfg):
    """Every battery kit must appear in tools/power_budget.py, so its budget is
    computed from one shared model rather than a hand-made README table."""
    import power_budget

    kit_key = cfg["__kit__"]
    mains = {"cn0549", "rpi5_hailo", "jetson_orin", "rpi5-hailo", "jetson-orin"}
    if kit_key in mains:
        pytest.skip("mains-powered kit")
    assert any(k.replace("_", "") == kit_key.replace("_", "").replace("-", "")
               for k in power_budget.KITS), (
        f"{kit_key} is not in power_budget.KITS"
    )
