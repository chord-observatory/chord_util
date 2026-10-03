"""Tools for RFI flagging.

This module contains the CHORD-specific tools for finding and removing Radio
Frequency Interference (RFI). The general, telescope-agnostic flagging
algorithms are in `draco` (`draco.util.rfi`, `draco.analysis.flagging`); the
pipeline tasks that apply them to CHORD data are in
`chord_pipeline.analysis.flagging`.

Masks are True where the data contain RFI (or are to be excluded) and False
elsewhere.

Static mask

- :py:data:`BAD_FREQUENCIES`
- :py:meth:`frequency_mask`

Times to treat with care

- :py:meth:`guess_fwhm`
- :py:meth:`daytime_flag`
- :py:meth:`transit_flag`
- :py:meth:`bright_source_flag`

Masks from the data

- :py:meth:`mask_1d`
- :py:meth:`mad_sumthreshold_mask`
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from caput.algorithms import median
from caput.astro import time as ctime
from caput.astro.skyfield import skyfield_wrapper, unix_to_skyfield_time

if TYPE_CHECKING:
    from caput.astro.observer import Observer

logger = logging.getLogger(__name__)

# Speed of light in m/s
_C = 299792458.0

# Ratio of the standard deviation to the median absolute deviation of a Gaussian
MAD_TO_RMS = 1.4826

# Ranges of bad frequencies, given by their start and end times (UNIX time, None
# for no limit) and their start and end frequencies (MHz).
#
# The "chord" list is the static mask made from the first CHORD data at DRAO,
# starting from CHIME's (ch_util.rfi.BAD_FREQUENCIES["chime"]). The bands
# commented out are CHIME bands not seen in the CHORD data.
BAD_FREQUENCIES = {
    "chord": [
        # Mobile services
        [[None, None], [356.0, 390.0]],
        # CHIME: bad bands at first light
        [[None, None], [449.41, 450.98]],
        [[None, None], [454.88, 456.05]],
        [[None, None], [457.62, 459.18]],
        [[None, None], [483.01, 485.35]],
        # UHF TV channel 17, broadcast from Omak
        [[None, None], [487.70, 494.34]],
        # UHF TV channel 19, broadcast from Omak
        # [[None, None], [497.85, 506.05]],
        # UHF TV channel 24, broadcast from Omak
        [[None, None], [529.10, 536.52]],
        # CHIME: narrow, high power bands, and sporadic RFI between them
        # [[None, None], [460.15, 460.55]],
        # [[None, None], [464.00, 470.32]],
        # CHIME: 6 MHz band
        # [[None, None], [505.85, 511.71]],
        # Bright band, present since CHIME's early days
        [[None, None], [517.97, 525.00]],
        # UHF TV channels 30 and 31, broadcast from Omak
        [[None, None], [564.65, 578.00]],
        # UHF TV channel 33
        [[None, None], [584.00, 590.00]],
        # UHF TV channel 35
        [[None, None], [596.00, 602.00]],
        # CHIME: low power band visible in long baselines
        # [[None, None], [602.00, 607.82]],
        # 600 MHz mobile bands: Rogers, Bell, Freedom
        [[None, None], [617.00, 627.00]],
        [[None, None], [627.00, 637.00]],
        [[None, None], [637.00, 642.00]],
        # Seen in the first CHORD data
        [[None, None], [642.00, 652.00]],
        # 700 and 800 MHz cellular (LTE / 5G NR) bands
        [[None, None], [717.0, 728.0]],  # supplemental downlink, n29
        [[None, None], [729.0, 745.12]],  # downlink, n12 / n85
        [[None, None], [746.0, 756.0]],  # downlink, n13
        [[None, None], [758.0, 768.0]],  # downlink, n14
        [[None, None], [824.0, 849.0]],  # uplink, n5
        [[None, None], [869.0, 894.0]],  # downlink, n5
        # Aircraft secondary surveillance radar / Mode S: 1030 MHz
        # interrogations and 1090 MHz replies (ADS-B)
        [[None, None], [1030.0, 1031.0]],
        [[None, None], [1089.0, 1092.0]],
        # Navigation and GNSS, and the 23 cm amateur band
        [[None, None], [1135.0, 1315.0]],
    ],
}


def frequency_mask(
    freq_centre: np.ndarray,
    freq_width: np.ndarray | float | None = None,
    timestamp: np.ndarray | float | None = None,
    instrument: str = "chord",
) -> np.ndarray:
    """Flag known bad frequencies.

    Parameters
    ----------
    freq_centre
        Centre of each frequency channel in MHz.
    freq_width
        Width of each frequency channel in MHz. If `None` (default), calculate
        the width from the separation of the channel centres. If an array, it
        must be broadcastable against `freq_centre`.
    timestamp
        UNIX observing time. If `None` (default), mask all the bands whatever
        their start and end times, otherwise mask only the bands in use at these
        times. If an array, it must be broadcastable against `freq_centre`.
    instrument
        Name of the instrument in :py:data:`BAD_FREQUENCIES`. Default "chord".

    Returns
    -------
    mask
        True for the bad frequency channels. The shape is the result of
        broadcasting `freq_centre` and `timestamp` together.
    """
    freq_centre = np.asarray(freq_centre)
    if freq_width is None:
        freq_width = np.abs(np.median(np.diff(freq_centre)))

    freq_start = freq_centre - freq_width / 2
    freq_end = freq_centre + freq_width / 2

    mask = np.zeros(np.broadcast(freq_centre, timestamp).shape, dtype=bool)

    try:
        bad_freq = BAD_FREQUENCIES[instrument]
    except KeyError as e:
        raise ValueError(f"No RFI flags defined for {instrument}") from e

    for (start_time, end_time), (fs, fe) in bad_freq:
        # A channel is bad if it overlaps the band
        fmask = (freq_end > fs) & (freq_start < fe)

        if timestamp is None:
            tmask = True
        else:
            tmask = np.ones_like(timestamp, dtype=bool)
            if start_time is not None:
                tmask &= timestamp >= start_time
            if end_time is not None:
                tmask &= timestamp <= end_time

        mask |= tmask & fmask

    return mask


def guess_fwhm(
    freq: float | np.ndarray,
    dec: float | None = None,
    sigma: bool = False,
    voltage: bool = False,
    seconds: bool = False,
    diameter: float = 6.0,
) -> float | np.ndarray:
    """Estimate the width of the primary beam of a CHORD dish.

    Assumes a diffraction-limited circular dish, whose power beam has a full
    width at half maximum of about 1.02 lambda / D.

    Parameters
    ----------
    freq
        Frequency in MHz.
    dec
        Declination of the source in radians. If given, divide by cos(dec) to
        convert from an angle on the sky to an hour angle.
    sigma
        Return the standard deviation of a Gaussian beam instead of the FWHM.
    voltage
        Return the width of the voltage beam instead of the power beam.
    seconds
        Return the time taken to drift through the width, in seconds, instead
        of degrees.
    diameter
        Dish diameter in metres. Default 6 m.

    Returns
    -------
    width
        Beam width in degrees, or seconds if `seconds` is True.
    """
    wavelength = _C / (np.asarray(freq, dtype=np.float64) * 1e6)
    width = np.degrees(1.02 * wavelength / diameter)

    if dec is not None:
        width = width / np.cos(dec)

    if seconds:
        width = width * 240.0 * ctime.SIDEREAL_S

    if voltage:
        width = width * np.sqrt(2.0)

    if sigma:
        width = width / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    return width


def _observer(obs: Observer | None) -> Observer:
    if obs is None:
        from chord_ephem.observers import chord as obs
    return obs


def daytime_flag(time: np.ndarray, obs: Observer | None = None) -> np.ndarray:
    """Flag the times between sunrise and sunset.

    Parameters
    ----------
    time : np.ndarray[ntime]
        UNIX times.
    obs : Observer, optional
        The observer. Default CHORD.

    Returns
    -------
    flag : np.ndarray[ntime]
        True for times during the day.
    """
    obs = _observer(obs)
    time = np.atleast_1d(time)
    flag = np.zeros(time.size, dtype=bool)

    for rise in obs.solar_rising(time.min() - 24.0 * 3600.0, time.max()):
        sets = obs.solar_setting(rise)[0]
        flag |= (time >= rise) & (time <= sets)

    return flag


def transit_flag(
    body,
    time: np.ndarray,
    nsigma: float = 2.0,
    freq: float | np.ndarray = 400.0,
    obs: Observer | None = None,
) -> np.ndarray:
    """Flag the times near the transit of a celestial body.

    Parameters
    ----------
    body : skyfield body
        Skyfield representation of a celestial body, e.g. from
        `chord_ephem.sources.source_dictionary`, or the Sun from
        `caput.astro.skyfield.skyfield_wrapper.ephemeris["sun"]`.
    time : np.ndarray[ntime]
        UNIX times.
    nsigma : float
        Number of beam sigmas to flag on either side of transit.
    freq : float or np.ndarray[nfreq]
        Frequency in MHz at which to evaluate the beam width.
    obs : Observer, optional
        The observer. Default CHORD.

    Returns
    -------
    flag : np.ndarray[ntime] or np.ndarray[nfreq, ntime]
        True for times within `nsigma` of a transit. Two dimensional if `freq`
        is an array.
    """
    obs = _observer(obs)
    time = np.atleast_1d(time)

    if np.isscalar(freq):
        flag = np.zeros(time.size, dtype=bool)
    else:
        freq = np.asarray(freq)
        flag = np.zeros((freq.size, time.size), dtype=bool)
        freq = freq[:, np.newaxis]

    transits = obs.transit_times(
        body, time[0] - 24.0 * 3600.0, time[-1] + 24.0 * 3600.0
    )

    for ttrans in transits:
        sf_time = unix_to_skyfield_time(ttrans)
        pos = obs.skyfield_obs().at(sf_time).observe(body)

        alt = pos.apparent().altaz()[0]
        dec = pos.cirs_radec(sf_time)[1]

        # Only if the body is above the horizon
        if alt.radians > 0.0:
            window = nsigma * guess_fwhm(
                freq, dec=dec.radians, sigma=True, seconds=True
            )
            flag |= (time >= ttrans - window) & (time <= ttrans + window)

    return flag


def bright_source_flag(
    time: np.ndarray,
    freq: np.ndarray,
    sources: tuple[str, ...] = ("CAS_A", "CYG_A", "TAU_A", "B0329+54"),
    nsigma_source: float = 1.0,
    nsigma_sun: float = 3.0,
    obs: Observer | None = None,
) -> np.ndarray:
    """Flag the times when the Sun or a bright source is transiting.

    Used to switch from SumThreshold to the MAD flagger around transits,
    where SumThreshold would remove real sky signal (as in
    `ch_pipeline.analysis.flagging.RFISensitivityMask`).

    Parameters
    ----------
    time : np.ndarray[ntime]
        UNIX times.
    freq : np.ndarray[nfreq]
        Frequencies in MHz.
    sources : tuple of str
        Names of sources in `chord_ephem.sources.source_dictionary`.
    nsigma_source, nsigma_sun : float
        Half-width of the flagged window around the transits of the sources
        and of the Sun, in beam sigmas.
    obs : Observer, optional
        The observer. Default CHORD.

    Returns
    -------
    flag : np.ndarray[nfreq, ntime]
        True within the transit windows.
    """
    from chord_ephem.sources import source_dictionary

    freq = np.asarray(freq)
    bodies = [(source_dictionary[src], nsigma_source) for src in sources]
    bodies.append((skyfield_wrapper.ephemeris["sun"], nsigma_sun))

    flag = np.zeros((freq.size, np.size(time)), dtype=bool)
    for body, nsigma in bodies:
        flag |= transit_flag(body, time, nsigma=nsigma, freq=freq, obs=obs)
    return flag


def mask_1d(
    y: np.ndarray,
    flag: np.ndarray,
    nsigma: float = 5.0,
    quantile: float = 0.15,
    win_f: int | None = 191,
) -> tuple[np.ndarray, np.ndarray]:
    """Flag whole frequency channels from a low quantile over time.

    The quantile over time of each channel is compared with its moving
    weighted median over `win_f` channels; channels deviating by more than
    `nsigma` times the (moving) MAD-derived standard deviation are flagged.
    This is the 1D stage of `draco.analysis.flagging.RFISensitivityMask`.

    Parameters
    ----------
    y : np.ndarray[nfreq, ntime]
        Data.
    flag : np.ndarray[nfreq, ntime]
        True for samples to ignore.
    nsigma : float
        Threshold in standard deviations.
    quantile : float
        Quantile over time to use, e.g. 0.15 to be robust to intermittent RFI.
    win_f : int or None
        Number of channels in the moving median. None for a single median
        over all frequencies.

    Returns
    -------
    flag_1d : np.ndarray[nfreq]
        True for flagged channels.
    y_static : np.ndarray[nfreq]
        The quantile over time of each channel, to subtract from the data.
    """
    y = np.ascontiguousarray(y, dtype=np.float64)
    w = np.ascontiguousarray(~flag, dtype=np.float64)

    medt_y = median.quantile(y, w, quantile)
    medt_w = np.any(w, axis=-1).astype(np.float64)

    def _median(x):
        if win_f is None:
            return median.weighted_median(x, medt_w)
        return median.moving_weighted_median(x, medt_w, win_f)

    absd = np.abs(medt_y - _median(medt_y))
    mad_1d = MAD_TO_RMS * _median(absd)

    return absd > nsigma * mad_1d, medt_y


def mad_sumthreshold_mask(
    y: np.ndarray,
    flag: np.ndarray,
    freq: np.ndarray,
    mask_type: str = "mad",
    madtimes: np.ndarray | None = None,
    nsigma_1d: float | None = 5.0,
    quantile_1d: float = 0.15,
    win_f_1d: int | None = 191,
    nsigma: float = 5.0,
    niter: int = 5,
    rho: float = 1.5,
    base_size: tuple[int, int] = (37, 181),
    mad_size: tuple[int, int] = (101, 31),
    tv_fraction: float | None = None,
    max_m: int = 64,
    sir_eta: float = 0.2,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[dict]]:
    """Make an RFI mask from a (freq, time) array with MAD and SumThreshold.

    The algorithm of `draco.analysis.flagging.RFISensitivityMask`, as a
    function of an array so that it can be applied to any data, e.g. the
    dish x RFI-monitor correlation:

    1. optionally flag channels with `mask_1d` and subtract their quantile
       over time;
    2. for thresholds decreasing as ``nsigma * rho**k`` (k = niter-1, ..., 0):
       estimate the background with a 2D moving median of size `base_size`,
       the noise with 1.4826 times the 2D moving median of the absolute
       residual over `mad_size`, and flag samples whose residual exceeds the
       threshold (MAD), or run SumThreshold on the residual, or both
       (`mask_type`).

    Parameters
    ----------
    y : np.ndarray[nfreq, ntime]
        Real data, e.g. |V| or |r|.
    flag : np.ndarray[nfreq, ntime]
        Initial flag (True = ignore): missing data and the static mask.
    freq : np.ndarray[nfreq]
        Frequencies in MHz.
    mask_type : {"mad", "sumthreshold", "combine"}
        "combine" uses MAD where `madtimes` is True and SumThreshold elsewhere,
        and extends the SumThreshold mask in time with SIR across those times.
    madtimes : np.ndarray[nfreq, ntime], optional
        Where to use MAD for "combine", e.g. from `bright_source_flag`.
    nsigma_1d, quantile_1d, win_f_1d
        Parameters of the 1D stage (`mask_1d`); `nsigma_1d=None` skips it.
    nsigma : float
        Final threshold in standard deviations.
    niter : int
        Number of iterations.
    rho : float
        Factor by which the threshold drops at each iteration.
    base_size, mad_size : (int, int)
        (channels, times) of the moving medians for the background and the
        noise.
    tv_fraction : float, optional
        If given, also flag whole 6 MHz TV channels where more than this
        fraction of the samples exceed the threshold
        (`draco.analysis.flagging.tv_channels_flag`). Default None (off).
    max_m : int
        Largest SumThreshold window.
    sir_eta : float
        Aggressiveness of SIR for "combine".
    return_stats : bool
        Also return the percentage flagged after each stage.

    Returns
    -------
    mask : np.ndarray[nfreq, ntime]
        True for flagged samples, including the initial `flag`.
    stats : list of dict
        If `return_stats`: for each stage, the percentage of samples that the
        stage would flag (``candidate``), newly flags (``added``) and the
        total flagged after it (``total``).
    """
    from draco.util import rfi as drfi

    if mask_type not in ("mad", "sumthreshold", "combine"):
        raise ValueError(f"Unknown mask_type {mask_type!r}")
    if mask_type == "combine" and madtimes is None:
        raise ValueError('mask_type "combine" needs madtimes')

    y = np.array(y, dtype=np.float64)
    current = np.array(flag, dtype=bool)
    stats = []

    def _record(stage, threshold, candidate):
        # Percentages flagged by a stage, before it is added to `current`
        return {
            "stage": stage,
            "threshold": threshold,
            "candidate": 100.0 * candidate.mean(),
            "added": 100.0 * (candidate & ~current).mean(),
        }

    stats.append(
        {
            "stage": "initial",
            "threshold": None,
            "candidate": 100.0 * current.mean(),
            "added": 100.0 * current.mean(),
            "total": 100.0 * current.mean(),
        }
    )

    if nsigma_1d is not None:
        flag_1d, y_static = mask_1d(y, current, nsigma_1d, quantile_1d, win_f_1d)
        flag_1d = np.broadcast_to(flag_1d[:, np.newaxis], y.shape)
        entry = _record("1d", nsigma_1d, flag_1d)
        current |= flag_1d
        y -= y_static[:, np.newaxis]
        entry["total"] = 100.0 * current.mean()
        stats.append(entry)

    for threshold in nsigma * rho ** np.arange(niter)[::-1]:
        background = median.moving_weighted_median(
            y, np.ascontiguousarray(~current, dtype=np.float64), base_size
        )
        dy = y - background
        ady = np.abs(dy)
        sigma = MAD_TO_RMS * median.moving_weighted_median(
            ady, np.ascontiguousarray(~current, dtype=np.float64), mad_size
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            ady_nsigma = np.where(sigma > 0, ady / sigma, 0.0)

        madmask = ady_nsigma > threshold
        tvmask = np.zeros_like(madmask)
        if tv_fraction is not None:
            from draco.analysis.flagging import tv_channels_flag

            tvmask = tv_channels_flag(ady_nsigma, freq, sigma=threshold, f=tv_fraction)
            madmask |= tvmask

        if mask_type == "mad":
            new = madmask
        else:
            stmask = drfi.sumthreshold(
                dy,
                max_m,
                start_flag=current | tvmask,
                threshold1=threshold,
                remove_median=False,
                correct_for_missing=True,
                rho=1.0,
                variance=sigma**2,
            )
            if mask_type == "sumthreshold":
                new = stmask
            else:
                new = np.where(madtimes, madmask, stmask)
                expanded = drfi.scale_invariant_rank(new, eta=sir_eta, axis=-1)
                new = np.where(madtimes, expanded, new)

        entry = _record(f"{mask_type} {threshold:.3g}", threshold, new)
        current |= new
        entry["total"] = 100.0 * current.mean()
        stats.append(entry)
        logger.info(
            "threshold %.3g: candidate %.2f%%, added %.2f%%, total %.2f%%",
            threshold,
            entry["candidate"],
            entry["added"],
            entry["total"],
        )

    if return_stats:
        return current, stats
    return current
