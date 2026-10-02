"""Tests for chord_util.rfi."""

import numpy as np
import pytest
from caput.astro.skyfield import skyfield_wrapper

from chord_util import rfi

# CHORD channels: 300 to 1500 MHz in 1200 / 6144 MHz steps
FREQ = np.linspace(300.0, 1500.0, 6145)
# 2026-09-11 16:29:18 PDT, the start of acq_20260911_232919_986789057
T0 = 1789169358.483057


def test_frequency_mask_bands():
    mask = rfi.frequency_mask(FREQ)
    assert mask.shape == FREQ.shape

    # Every channel overlapping a band is masked, and nothing else
    df = np.median(np.diff(FREQ))
    expected = np.zeros_like(mask)
    for _, (fs, fe) in rfi.BAD_FREQUENCIES["chord"]:
        expected |= (FREQ + df / 2 > fs) & (FREQ - df / 2 < fe)
    assert np.array_equal(mask, expected)

    assert mask[np.argmin(np.abs(FREQ - 1200.0))]  # GNSS
    assert mask[np.argmin(np.abs(FREQ - 600.0))]  # TV channel 35
    assert not mask[np.argmin(np.abs(FREQ - 1420.4))]  # HI
    assert not mask[np.argmin(np.abs(FREQ - 950.0))]
    assert 0.25 < mask.mean() < 0.35


def test_frequency_mask_time(monkeypatch):
    bands = [[[None, 100.0], [400.0, 410.0]], [[200.0, None], [500.0, 510.0]]]
    monkeypatch.setitem(rfi.BAD_FREQUENCIES, "test", bands)
    freq = np.array([405.0, 505.0, 700.0])

    # All bands without a time
    assert rfi.frequency_mask(freq, instrument="test").tolist() == [True, True, False]
    # Only the bands in use at the time
    assert rfi.frequency_mask(freq, timestamp=50.0, instrument="test").tolist() == [
        True,
        False,
        False,
    ]
    assert rfi.frequency_mask(freq, timestamp=150.0, instrument="test").tolist() == [
        False,
        False,
        False,
    ]
    assert rfi.frequency_mask(freq, timestamp=250.0, instrument="test").tolist() == [
        False,
        True,
        False,
    ]

    # Broadcast against an array of times
    mask = rfi.frequency_mask(
        freq[:, None],
        freq_width=0.2,
        timestamp=np.array([50.0, 250.0]),
        instrument="test",
    )
    assert mask.tolist() == [[True, False], [False, True], [False, False]]

    with pytest.raises(ValueError, match="No RFI flags"):
        rfi.frequency_mask(freq, instrument="not_an_instrument")


def test_guess_fwhm():
    # 1.02 lambda / D at 600 MHz for a 6 m dish
    fwhm = rfi.guess_fwhm(600.0)
    assert fwhm == pytest.approx(np.degrees(1.02 * 0.49965 / 6.0), rel=1e-4)
    assert rfi.guess_fwhm(1200.0) == pytest.approx(fwhm / 2)
    assert rfi.guess_fwhm(600.0, sigma=True) == pytest.approx(fwhm / 2.35482, rel=1e-5)
    assert rfi.guess_fwhm(600.0, voltage=True) == pytest.approx(fwhm * np.sqrt(2))
    assert rfi.guess_fwhm(600.0, dec=np.radians(60.0)) == pytest.approx(2 * fwhm)
    # One degree drifts by in 4 sidereal minutes
    assert rfi.guess_fwhm(600.0, seconds=True) == pytest.approx(
        fwhm * 239.345, rel=1e-4
    )
    assert rfi.guess_fwhm(FREQ).shape == FREQ.shape


def test_daytime_flag():
    # Local midnight, morning, afternoon of 2026-09-11/12 (PDT = UTC-7)
    hours = np.array([0.0, 4.0, 10.0, 15.0, 22.0])
    time = 1789110000.0 + 3600.0 * hours  # 2026-09-11 00:00 PDT
    assert rfi.daytime_flag(time).tolist() == [False, False, True, True, False]


def test_transit_flag():
    from chord_ephem.observers import chord
    from chord_ephem.sources import CygA

    time = T0 + 10.0 * np.arange(4645)
    ttrans = chord.transit_times(CygA, time[0], time[-1])[0]

    flag = rfi.transit_flag(CygA, time, nsigma=2.0, freq=600.0)
    assert flag.shape == time.shape
    window = 2.0 * rfi.guess_fwhm(
        600.0, dec=np.radians(40.73), sigma=True, seconds=True
    )
    flagged = time[flag]
    assert flagged.min() == pytest.approx(ttrans - window, abs=11.0)
    assert flagged.max() == pytest.approx(ttrans + window, abs=11.0)

    # Wider at lower frequencies
    flag2d = rfi.transit_flag(CygA, time, freq=np.array([400.0, 800.0]))
    assert flag2d.shape == (2, time.size)
    assert flag2d[0].sum() > flag2d[1].sum() > 0

    # The Sun does not transit during the night
    sun = skyfield_wrapper.ephemeris["sun"]
    assert not rfi.transit_flag(sun, time).any()
