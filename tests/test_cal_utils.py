"""Tests for chord_util.cal_utils."""

import numpy as np
import pytest

from chord_util import cal_utils

HA = np.linspace(-4.0, 4.0, 161)
SIGMA = 1.2  # width of the transit in degrees of hour angle


def _transit(peak=(1.0, 2.0, 0.5), centre=(0.0, 0.3, -0.2), noise=1e-3, seed=0):
    """Response of a few inputs to a point source transit.

    A Gaussian beam in amplitude and a linear phase, with Gaussian noise.
    """
    rng = np.random.default_rng(seed)
    peak = np.asarray(peak)[:, np.newaxis]
    centre = np.asarray(centre)[:, np.newaxis]
    amp = peak * np.exp(-0.5 * ((HA - centre) / SIGMA) ** 2)
    phase = 0.4 + 0.05 * HA[np.newaxis, :] * np.arange(1, len(peak) + 1)[:, None]
    resp = amp * np.exp(1j * phase)
    err = np.full(resp.shape, noise)
    resp = resp + noise * (
        rng.standard_normal(resp.shape) + 1j * rng.standard_normal(resp.shape)
    )
    return resp, err, centre[:, 0]


def test_poly_log_amp_poly_phase_fit():
    resp, err, centre = _transit()
    fit = cal_utils.FitPolyLogAmpPolyPhase(poly_deg_amp=5, poly_deg_phi=5)
    # All hour angles, as in the CHORD configs (moving_window: false)
    fit.fit(HA, resp, err, width=SIGMA, absolute_sigma=True, moving_window=None)

    assert fit.param.shape == (3, fit.nparam)
    assert np.isfinite(fit.param).all()
    np.testing.assert_allclose(fit.peak(), centre, atol=0.02)

    near = np.abs(HA) < 2.0
    model = fit.predict(HA)
    np.testing.assert_allclose(model[:, near], resp[:, near], atol=0.01)


def test_gauss_amp_poly_phase_fit():
    resp, err, centre = _transit()
    fit = cal_utils.FitGaussAmpPolyPhase(poly_deg_phi=3)
    fit.fit(HA, resp, err, width=SIGMA, absolute_sigma=True)

    names = [n.decode() for n in fit.parameter_names]
    # The sign of the amplitude can be traded for a phase of pi
    np.testing.assert_allclose(np.abs(fit.param[:, 0]), [1.0, 2.0, 0.5], rtol=0.01)
    np.testing.assert_allclose(fit.predict(HA), resp, atol=0.01)
    np.testing.assert_allclose(fit.param[:, names.index("centroid")], centre, atol=0.02)
    fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * SIGMA
    np.testing.assert_allclose(fit.param[:, names.index("fwhm")], fwhm, rtol=0.01)


def test_failed_fit_is_skipped(monkeypatch):
    """An input whose fit fails ("No peak found") is left as NaN, not raised."""
    resp, err, _ = _transit()

    def no_peak(self, param=None):
        param = self.param if param is None else param
        return np.full(param.shape[:-1], np.nan)

    monkeypatch.setattr(cal_utils.FitPolyLogAmpPolyPhase, "peak", no_peak)
    fit = cal_utils.FitPolyLogAmpPolyPhase()
    fit.fit(HA, resp, err, width=SIGMA, absolute_sigma=True, moving_window=None)
    assert np.isnan(fit.param).all()


def test_flag_outliers():
    rng = np.random.default_rng(1)
    raw = 1.0 + 0.01 * rng.standard_normal(500)
    raw[[100, 300]] += 1.0
    flag = np.ones(raw.size, dtype=bool)
    out = cal_utils.flag_outliers(raw, flag, window=25, nsigma=5.0)
    assert not out[[100, 300]].any()
    assert out.sum() >= raw.size - 10


def test_interpolate_gain():
    freq = np.linspace(400.0, 800.0, 401)
    gain = (1.0 + 0.2 * np.sin(freq / 40.0)) * np.exp(1j * freq / 100.0)
    gain = np.stack([gain, 2.0 * gain], axis=1)
    weight = np.full(gain.shape, 1e6)
    gap = (freq > 600.0) & (freq < 604.0)
    weight[gap] = 0.0
    weight[:, 1] = 0.0  # no valid data for the second input

    ginterp, winterp = cal_utils.interpolate_gain_quiet(freq, gain, weight)

    # Unflagged gains are kept, flagged ones are interpolated
    np.testing.assert_array_equal(ginterp[~gap, 0], gain[~gap, 0])
    np.testing.assert_allclose(ginterp[gap, 0], gain[gap, 0], atol=0.01)
    assert (winterp[gap, 0] > 0).all()
    # An input with no valid gains is zeroed
    assert (ginterp[:, 1] == 0).all() and (winterp[:, 1] == 0).all()


@pytest.mark.parametrize("length_scale", [10.0, 30.0])
def test_interpolate_gain_real(length_scale):
    freq = np.linspace(400.0, 800.0, 201)
    gain = (1.0 + 0.1 * np.cos(freq / 30.0))[:, np.newaxis]
    weight = np.full(gain.shape, 1e6)
    weight[100:103] = 0.0
    ginterp, _ = cal_utils.interpolate_gain_quiet(
        freq, gain, weight, length_scale=length_scale
    )
    assert np.isrealobj(ginterp) or np.all(ginterp.imag == 0)
    np.testing.assert_allclose(ginterp[100:103, 0], gain[100:103, 0], atol=0.01)
